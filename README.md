# 🏠 MLOps - Prédiction Prix Maisons (Version GitHub Optimisée)

> **Système MLOps complet pour prédire le prix des maisons avec Linear Regression**  
> Version optimisée pour GitHub - 95% d'économie d'espace (50MB → 2MB)

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104-green.svg)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://www.docker.com/)
[![MLOps](https://img.shields.io/badge/MLOps-Complete-orange.svg)](https://ml-ops.org/)

## 🎯 **Objectifs du Projet**

Ce projet implémente un pipeline MLOps complet pour la prédiction de prix immobiliers, respectant toutes les bonnes pratiques industrielles :

- ✅ **Pipeline ML complet** avec preprocessing, entraînement et validation
- ✅ **API REST** pour les prédictions en temps réel avec FastAPI
- ✅ **Configuration Management** avec environnements (dev/prod)
- ✅ **Docker & Containerisation** pour le déploiement reproductible
- ✅ **Logging structuré** pour l'audit et monitoring (exigence projet)
- ✅ **Bonnes pratiques MLOps** industrielles

## 📊 **Performance du Modèle**

| Métrique | Valeur | Description |
|----------|--------|-------------|
| **R² test** | **0.4924** | Coefficient de détermination |
| **RMSE test** | **253,409$** | Erreur quadratique moyenne |
| **MAE test** | **163,731$** | Erreur absolue moyenne |
| **Généralisation** | **0.0231** | Écart validation/test (excellent) |
| **Temps prédiction** | **~15ms** | Temps de réponse API |

## 🏗️ **Architecture MLOps**

```
📦 prediction_prix_maison/
├── 🔧 config/                    # Configuration centralisée YAML
│   ├── environments/             # Environnements (dev/prod)
│   │   ├── development.yaml      # Config développement
│   │   └── production.yaml       # Config production
│   └── models/                   # Configuration modèles
│       └── linear_regression.yaml
├── 📊 src/
│   ├── data/                     # Pipeline preprocessing
│   │   └── preprocesseur.py      # StandardScaler + nettoyage
│   ├── models/                   # Entraînement ML
│   │   ├── entraineur_optimized.py  # Version GitHub (Linear Regression)
│   │   └── entraineur_avec_config.py # Version complète (3 modèles)
│   ├── api/                      # API REST FastAPI
│   │   └── app.py                # API avec logging structuré
│   └── config/                   # Gestionnaire configuration
│       └── gestionnaire_config.py
├── 🐳 Dockerfile                 # Containerisation optimisée
├── 📋 docker-compose.yml         # Orchestration services
├── ⚙️ config_cli.py              # CLI gestion configuration
├── 📝 tests/                     # Tests unitaires
├── 📊 models/                    # Modèles entraînés (optimisés)
│   ├── preprocesseur.pkl         # Pipeline preprocessing (2KB)
│   └── modele_linear_regression.pkl # Modèle optimisé (1.7KB)
└── 📋 requirements.txt           # Dépendances Python
```

## 🚀 **Quick Start**

### **1. Installation**
```bash
# Cloner le projet
git clone https://github.com/votre-username/prediction_prix_maison.git
cd prediction_prix_maison

# Créer environnement virtuel
python -m venv venv_mlops
venv_mlops\Scripts\activate     # Windows
# ou
source venv_mlops/bin/activate  # Linux/Mac

# Installer dépendances
pip install -r requirements.txt
```

### **2. Préparation des données**
```bash
# Générer le preprocesseur optimisé
python src/data/preprocesseur.py
```

### **3. Entraînement du modèle**
```bash
# Entraîner Linear Regression (le meilleur modèle)
python src/models/entraineur_optimized.py --environment development

# Résultat attendu:
# 🏆 R² test: 0.4924
# 🎯 RMSE test: 253409.77
# 💾 Modèle sauvé: modele_linear_regression_development.pkl
```

### **4. Lancer l'API**
```bash
# Démarrer l'API REST
python src/api/app.py

# API disponible sur: http://localhost:8000
# Documentation Swagger: http://localhost:8000/docs
```

### **5. Test de prédiction**
```bash
# Exemple de prédiction (PowerShell)
Invoke-RestMethod -Uri "http://localhost:8000/predict" -Method POST -ContentType "application/json" -Body '{
  "bedrooms": 3,
  "bathrooms": 2,
  "sqft_living": 1800,
  "sqft_lot": 7500,
  "floors": 2,
  "waterfront": 0,
  "view": 0,
  "condition": 3,
  "sqft_above": 1800,
  "sqft_basement": 0,
  "yr_built": 1995
}'

# Résultat attendu:
# {
#   "prix_predit": 424413.27,
#   "confiance": "Bonne",
#   "modele_utilise": "linear_regression",
#   "duration_ms": 15.2
# }
```

## 🐳 **Déploiement Docker**

### **Build et Run**
```bash
# Build de l'image optimisée
docker build -t mlops-prix-maison .

# Lancer le container
docker run -d -p 8000:8000 --name mlops-api mlops-prix-maison

# Tester l'API containerisée
curl http://localhost:8000/health
```

### **Docker Compose (complet)**
```bash
# Lancer tous les services
docker-compose up -d

# Services disponibles:
# - API MLOps: http://localhost:8000
# - MLflow UI: http://localhost:5000 (optionnel)
```

## ⚙️ **Configuration Management**

### **CLI de Configuration**
```bash
# Valider configuration
python config_cli.py validate --environment development

# Afficher configuration détaillée
python config_cli.py show --environment production --verbose

# Comparer environnements
python config_cli.py diff --env1 development --env2 production

# Lister modèles disponibles
python config_cli.py list-models
```

### **Environnements Disponibles**
- **`development`** : Debug activé, reload automatique, logs détaillés
- **`production`** : Optimisé performance, logs minimaux, sécurisé

### **Configuration Modifiable Sans Code**
```yaml
# config/environments/development.yaml
models:
  enabled_models: ["linear_regression"]  # Modèles actifs

api:
  port: 8000
  log_level: "debug"

data:
  features: [
    "bedrooms", "bathrooms", "sqft_living", # 11 features optimales
    "sqft_lot", "floors", "waterfront", "view",
    "condition", "sqft_above", "sqft_basement", "yr_built"
  ]
```

## 📝 **Logging Structuré (Exigence Projet)**

### **Format des logs respectant les exigences**
```json
{
  "timestamp": "2025-07-24T19:55:00Z",          // ✅ timestamp de requête
  "request_id": "abc-123-def",
  "event_type": "prediction",
  "input": {
    "features": {                               // ✅ entrées (features)
      "bedrooms": 3,
      "bathrooms": 2,
      "sqft_living": 1800,
      "sqft_lot": 7500,
      "floors": 2,
      "waterfront": 0,
      "view": 0,
      "condition": 3,
      "sqft_above": 1800,
      "sqft_basement": 0,
      "yr_built": 1995
    },
    "feature_count": 11
  },
  "output": {
    "prediction": 424413.27,                   // ✅ prédiction
    "confidence": "Bonne",
    "model_used": "linear_regression"
  },
  "performance": {
    "duration_ms": 15.2                        // ✅ durée
  },
  "status": "success",
  "client_ip": "127.0.0.1"
}
// ✅ Permet audit et monitoring complet
```

### **Endpoints de Monitoring**
- **`/health`** : Santé de l'API et du modèle
- **`/debug`** : Informations techniques détaillées
- **`/docs`** : Documentation Swagger interactive

### **Fichiers de logs**
```bash
# Voir les logs structurés
Get-Content reports/logs/api_audit.log -Tail 10
```

## 🧪 **Tests**

### **Tests Unitaires**
```bash
# Installer dépendances test
pip install pytest pytest-asyncio httpx

# Lancer tests complets
pytest tests/ -v

# Tests de performance
python tests/test_performance.py
```

### **Tests API**
```bash
# Health check
curl http://localhost:8000/health

# Test prédiction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"bedrooms": 3, "bathrooms": 2, ...}'

# Documentation interactive
# Ouvrir: http://localhost:8000/docs
```

## 📊 **Optimisations GitHub**

### **🎯 Pourquoi cette version optimisée ?**

| Aspect | Avant | Après | Économie |
|--------|-------|-------|----------|
| **Taille totale** | ~50MB | ~2MB | **95%** |
| **Modèles** | 3 algorithmes | 1 (le meilleur) | **66%** |
| **Performance** | R² = 0.4924 | R² = 0.4924 | **0% impact** |
| **Compatibilité GitHub** | ❌ Trop lourd | ✅ Compatible | **100%** |

### **🤖 Modèles - Avant/Après optimisation**

| Modèle | Taille | R² test | Statut |
|--------|--------|---------|--------|
| Random Forest | 1.5MB | 0.4653 | ❌ Supprimé |
| Gradient Boosting | 1.2MB | 0.4552 | ❌ Supprimé |
| **Linear Regression** | **1.7KB** | **0.4924** | ✅ **Gardé (MEILLEUR)** |

### **📈 Impact sur les performances**
- ✅ **Aucun impact négatif** - Meilleur modèle conservé
- ✅ **API plus rapide** - Moins de modèles = chargement instantané
- ✅ **Déploiement simplifié** - Configuration unique
- ✅ **Maintenance réduite** - Moins de complexité

## 🎯 **Fonctionnalités Principales**

### **✅ Pipeline ML Complet**
- **Preprocessing automatique** avec StandardScaler et gestion valeurs manquantes
- **Validation croisée** 5-fold pour robustesse
- **Métriques complètes** : R², RMSE, MAE, MAPE
- **Tracking MLflow** optionnel pour expérimentations

### **✅ API REST Professionnelle**
- **FastAPI** avec validation Pydantic automatique
- **Documentation Swagger** générée automatiquement
- **Gestion d'erreurs** robuste avec codes HTTP appropriés
- **Headers de performance** (temps de réponse, request ID)
- **Logging structuré** pour audit complet

### **✅ Configuration Management**
- **Fichiers YAML** pour chaque environnement
- **CLI complet** pour validation et gestion
- **Variables d'environnement** pour déploiement
- **Hyperparamètres modifiables** sans toucher au code

### **✅ Logging & Monitoring**
- **Logs JSON structurés** respectant les exigences projet
- **Audit complet** des prédictions avec timestamp, features, résultats
- **Métriques de performance** en temps réel
- **Format standardisé** pour intégration monitoring

### **✅ Docker & Déploiement**
- **Image multi-stage** optimisée (builder + runtime)
- **Utilisateur non-root** pour sécurité
- **Health checks** automatiques intégrés
- **Variables d'environnement** pour configuration

## 📈 **Métriques de Projet**

| Composant | Status | Performance | Détails |
|-----------|--------|-------------|---------|
| **Data Pipeline** | ✅ | 4600 lignes (0% perte) | Preprocessing complet |
| **Model Training** | ✅ | 0.008s | Ultra-rapide Linear Regression |
| **API Response** | ✅ | ~15ms | Temps de réponse excellent |
| **Docker Build** | ✅ | ~2min | Build optimisé multi-stage |
| **Memory Usage** | ✅ | <100MB | Runtime très léger |
| **Model Accuracy** | ✅ | R² = 0.4924 | Performance maintenue |

## 🛠️ **Technologies Utilisées**

### **Machine Learning**
- ![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-orange.svg) Algorithmes ML
- ![pandas](https://img.shields.io/badge/pandas-2.0-blue.svg) Manipulation données
- ![numpy](https://img.shields.io/badge/numpy-1.24-blue.svg) Calculs numériques

### **API & Web**
- ![FastAPI](https://img.shields.io/badge/FastAPI-0.104-green.svg) Framework API moderne
- ![uvicorn](https://img.shields.io/badge/uvicorn-0.23-green.svg) Serveur ASGI
- ![pydantic](https://img.shields.io/badge/pydantic-2.4-green.svg) Validation données

### **Configuration & CLI**
- ![PyYAML](https://img.shields.io/badge/PyYAML-6.0-red.svg) Configuration YAML
- ![click](https://img.shields.io/badge/click-8.1-blue.svg) Interface ligne commande

### **Containerisation**
- ![Docker](https://img.shields.io/badge/Docker-24.0-blue.svg) Containerisation
- ![docker-compose](https://img.shields.io/badge/docker--compose-2.20-blue.svg) Orchestration

### **Monitoring & Tracking**
- ![MLflow](https://img.shields.io/badge/MLflow-2.8-orange.svg) Tracking expériences (optionnel)
- **Logging JSON** structuré pour audit

### **Tests & Qualité**
- ![pytest](https://img.shields.io/badge/pytest-7.4-green.svg) Tests unitaires
- ![httpx](https://img.shields.io/badge/httpx-0.24-blue.svg) Tests API

## 📋 **Prérequis**

- **Python 3.11+** (recommandé 3.11.5)
- **Docker** (optionnel, pour containerisation)
- **4GB RAM** minimum
- **100MB espace disque** (version optimisée)
- **Windows 10/11**, **Linux**, ou **macOS**

## 🔧 **Structure des Fichiers Optimisée**

```
prediction_prix_maison/          # 📦 2MB total (optimisé)
├── README.md                   # 📝 Documentation complète
├── requirements.txt            # 📋 Dépendances (minimal)
├── Dockerfile                  # 🐳 Image optimisée
├── docker-compose.yml          # 🐳 Orchestration
├── .gitignore                  # 🚫 Exclusions GitHub
├── config_cli.py               # ⚙️ CLI gestion config
│
├── config/                     # ⚙️ Configuration centralisée
│   ├── environments/
│   │   ├── development.yaml    # 🔧 Config dev
│   │   └── production.yaml     # 🏭 Config prod
│   └── models/
│       └── linear_regression.yaml # 🤖 Config modèle
│
├── src/                        # 💻 Code source
│   ├── api/
│   │   └── app.py              # 🌐 API FastAPI complète
│   ├── config/
│   │   └── gestionnaire_config.py # ⚙️ Gestionnaire config
│   ├── data/
│   │   └── preprocesseur.py    # 📊 Pipeline données
│   └── models/
│       ├── entraineur_optimized.py    # 🏆 Version GitHub
│       └── entraineur_avec_config.py  # 📈 Version complète
│
├── models/                     # 🤖 Modèles (optimisés)
│   ├── preprocesseur.pkl       # 📊 2KB - Pipeline preprocessing
│   └── modele_linear_regression.pkl # 🏆 1.7KB - Meilleur modèle
│
├── data/                       # 📊 Données (ignoré par Git)
│   ├── raw/                    # 📥 Données brutes
│   └── processed/              # ⚙️ Données preprocessées
│
├── tests/                      # 🧪 Tests unitaires
│   ├── test_api.py            # 🌐 Tests API
│   └── test_performance.py    # ⚡ Tests performance
│
└── reports/                    # 📋 Rapports et logs
    └── logs/                   # 📝 Logs structurés JSON
        └── api_audit.log       # 📊 Audit complet
```

## 🤝 **Contribution**

1. **Fork** le projet
2. Créer une branche feature (`git checkout -b feature/amazing-feature`)
3. **Commit** les changements (`git commit -m 'Add amazing feature'`)
4. **Push** vers la branche (`git push origin feature/amazing-feature`)
5. Ouvrir une **Pull Request**

### **Standards de contribution**
- Code Python avec **type hints**
- Tests unitaires pour nouvelles fonctionnalités
- Documentation mise à jour
- Respect des conventions **PEP 8**

## 📜 **Licence**

Ce projet est sous licence MIT. Voir le fichier [`LICENSE`](LICENSE) pour plus de détails.

## 👥 **Auteurs**

- **MLOps Team** - Implémentation complète du pipeline
- **Version GitHub Optimisée** - Optimisation pour partage et déploiement

## 🎯 **Roadmap - Prochaines Étapes**

### **Phase 3 - Automatisation**
- [ ] **CI/CD** avec GitHub Actions
- [ ] **Tests automatisés** sur pull requests
- [ ] **Déploiement automatique** vers staging/production

### **Phase 4 - Monitoring Avancé**
- [ ] **Monitoring** avec Prometheus + Grafana
- [ ] **Alerting** sur dégradation performance
- [ ] **Dashboard** métriques temps réel

### **Phase 5 - Déploiement Cloud**
- [ ] **AWS/Azure/GCP** deployment
- [ ] **Kubernetes** orchestration
- [ ] **Auto-scaling** selon la charge

## 📊 **Comparaison Versions**

| Fonctionnalité | Version Complète | Version GitHub | Impact |
|---------------|------------------|----------------|--------|
| **Algorithmes ML** | 3 (Linear, RF, GB) | 1 (Linear) | Performance identique |
| **Taille projet** | ~50MB | ~2MB | 95% économie |
| **Temps build Docker** | ~5min | ~2min | 60% plus rapide |
| **Complexité maintenance** | Élevée | Réduite | Simplification |
| **Performance R²** | 0.4924 | 0.4924 | Identique |
| **Compatibilité GitHub** | ❌ | ✅ | Compatible |

## 🏆 **Achievements MLOps**

- ✅ **Pipeline complet** de bout en bout
- ✅ **API production-ready** avec documentation
- ✅ **Configuration management** professionnel
- ✅ **Logging structuré** respectant les exigences
- ✅ **Containerisation** optimisée
- ✅ **Tests automatisés** pour la robustesse
- ✅ **Documentation complète** pour maintenance
- ✅ **Optimisation GitHub** sans perte performance

---

> **🎯 Note importante** : Cette version est spécialement optimisée pour GitHub avec 95% d'économie d'espace.  
> **Performance identique** (meilleur modèle gardé) mais taille réduite de 50MB à 2MB.  
> **Idéal pour** : partage, présentation, et déploiements légers.

---

## 📞 **Support & Contact**

- **Issues** : [GitHub Issues](https://github.com/votre-username/prediction_prix_maison/issues)
- **Documentation** : README.md (ce fichier)
- **API Documentation** : http://localhost:8000/docs (quand l'API est lancée)

**⭐ N'hésitez pas à mettre une étoile si ce projet vous aide !**