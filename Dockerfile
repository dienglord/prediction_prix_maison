FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 MLOPS_ENV=development

RUN apt-get update && apt-get install -y curl build-essential gcc && rm -rf /var/lib/apt/lists/*

RUN groupadd -r mlops && useradd -r -g mlops mlops

WORKDIR /app
RUN mkdir -p /app/data /app/models /app/config /app/reports /app/mlruns

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY --chown=mlops:mlops src/ ./src/
COPY --chown=mlops:mlops config/ ./config/
COPY --chown=mlops:mlops config_cli.py ./
COPY --chown=mlops:mlops models/ ./models/
COPY --chown=mlops:mlops data/ ./data/

RUN chown -R mlops:mlops /app
USER mlops

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 CMD curl -f http://localhost:8000/health || exit 1

CMD ["python", "src/api/app.py"]