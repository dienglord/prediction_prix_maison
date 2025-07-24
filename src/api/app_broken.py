import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import json
import time
from datetime import datetime
import traceback
from typing import Dict, Any, Optional
import uuid
import os
import sys

# Configuration du logging structuré
class StructuredLogger:
    """Gestionnaire de logging structuré pour l'audit MLOps."""
    
    def __init__(self, log_file: str = "reports/logs/api_audit.log"):
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Configuration du logger principal
        self.logger = logging.getLogger("mlops_api")
        self.logger.setLevel(logging.INFO)
        
        # Éviter les doublons de handlers
        if not self.logger.handlers:
            # Handler pour fichier avec format JSON
            file_handler = logging.FileHandler(self.log_file, encoding='utf-8')
            file_handler.setLevel(logging.INFO)
            
            # Handler pour console
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(logging.INFO)
            
            # Format structuré JSON
            formatter = logging.Formatter('%(message)s')
            file_handler.setFormatter(formatter)
            console_handler.setFormatter(formatter)
            
            self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)
    
    def log_prediction(self, 
                      request_id: str,
                      features: Dict[str, Any],
                      prediction: float,
                      confidence: str,
                      model_used: str,
                      duration_ms: float,
                      client_ip: str = None,
                      error: str = None):
        """Log une prédiction avec format structuré."""
        
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "request_id": request_id,
            "event_type": "prediction",
            "client_ip": client_ip,
            "input": {
                "features": features,
                "feature_count": len(features)
            },
            "output": {
                "prediction": prediction,
                "confidence": confidence,
                "model_used": model_used
            },
            "performance": {
                "duration_ms": round(duration_ms, 2)
            },
            "status": "success" if error is None else "error",
            "error": error,
            "metadata": {
                "api_version": "2.0",
                "environment": os.getenv("MLOPS_ENV", "development")
            }
        }
        
        # Log en format JSON
        self.logger.info(json.dumps(log_entry, ensure_ascii=False))
    
    def log_health_check(self, status: str, model_loaded: bool, duration_ms: float):
        """Log un health check."""
        
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "event_type": "health_check",
            "status": status,
            "model_loaded": model_loaded,
            "duration_ms": round(duration_ms, 2),
            "environment": os.getenv("MLOPS_ENV", "development")
        }
        
        self.logger.info(json.dumps(log_entry, ensure_ascii=False))
    
    def log_error(self, error_type: str, message: str, details: Dict = None):
        """Log une erreur système."""
        
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "event_type": "error",
            "error_type": error_type,
            "message": message,
            "details": details or {},
            "environment": os.getenv("MLOPS_ENV", "development")
        }
        
        self.logger.error(json.dumps(log_entry, ensure_ascii=False))

# Initialisation du logger structuré
structured_logger = StructuredLogger()

# Modèles Pydantic
class MaisonFeatures(BaseModel):
    """Modèle de validation pour les features d'une maison."""
    bedrooms: int = Field(..., ge=0, le=20, description="Nombre de chambres")
    bathrooms: float = Field(..., ge=0, le=10, description="Nombre de salles de bain")
    sqft_living: int = Field(..., ge=100, le=20000, description="Surface habitable en pieds carrés")
    sqft_lot: int = Field(..., ge=500, le=100000, description="Surface du terrain en pieds carrés")
    floors: float = Field(..., ge=1, le=5, description="Nombre d'étages")
    waterfront: int = Field(..., ge=0, le=1, description="Vue sur l'eau (0/1)")
    view: int = Field(..., ge=0, le=4, description="Qualité de la vue (0-4)")
    condition: int = Field(..., ge=1, le=5, description="Condition de la maison (1-5)")
    sqft_above: int = Field(..., ge=0, le=20000, description="Surface au-dessus du sol")
    sqft_basement: int = Field(..., ge=0, le=5000, description="Surface du sous-sol")
    yr_built: int = Field(..., ge=1900, le=2025, description="Année de construction")

class PredictionResponse(BaseModel):
    """Modèle de réponse pour une prédiction."""
    request_id: str
    prix_predit: float
    confiance: str
    modele_utilise: str
    timestamp: str
    duration_ms: float

class HealthResponse(BaseModel):
    """Modèle de réponse pour le health check."""
    status: str
    modele_charge: str
    preprocesseur_charge: bool
    timestamp: str
    duration_ms: float
    environment: str

# Application FastAPI
app = FastAPI(
    title="🏠 API Prédiction Prix Maisons - MLOps",
    description="API REST pour prédire le prix des maisons avec logging structuré complet",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Variables globales pour le modèle
modele_charge = None
preprocesseur_charge = None
nom_modele = "Erreur"

def charger_modele_et_preprocesseur():
    """Charge le modèle et le preprocesseur avec gestion d'erreurs."""
    global modele_charge, preprocesseur_charge, nom_modele
    
    try:
        structured_logger.logger.info("🔍 === CHARGEMENT MODÈLE AVEC LOGGING ===")
        
        # Chemin des modèles
        models_dir = Path("models")
        structured_logger.logger.info(f"Dossier models: {models_dir}")
        structured_logger.logger.info(f"Existe: {models_dir.exists()}")
        
        if models_dir.exists():
            files = list(models_dir.glob("*"))
            structured_logger.logger.info(f"Fichiers dans models/: {[f.name for f in files]}")
        
        # Charger le preprocesseur
        preprocesseur_path = models_dir / "preprocesseur.pkl"
        if not preprocesseur_path.exists():
            raise FileNotFoundError(f"Preprocesseur non trouvé: {preprocesseur_path}")
        
        # CORRECTION : Charger directement le preprocesseur
        preprocesseur_charge = joblib.load(preprocesseur_path)
        structured_logger.logger.info(f"✅ Preprocesseur chargé: {type(preprocesseur_charge)}")
        
        # Trouver le meilleur modèle
        modeles_disponibles = list(models_dir.glob("modele_*.pkl"))
        if not modeles_disponibles:
            raise FileNotFoundError("Aucun modèle trouvé dans models/")
        
        structured_logger.logger.info(f"Modèles disponibles: {[m.name for m in modeles_disponibles]}")
        
        # Charger le premier modèle trouvé
        modele_path = modeles_disponibles[0]
        donnees_modele = joblib.load(modele_path)
        
        # CORRECTION : Extraire le modèle du dictionnaire si nécessaire
        if isinstance(donnees_modele, dict):
            modele_charge = donnees_modele.get('modele', donnees_modele)
            nom_modele = donnees_modele.get('nom_modele', modele_path.stem.replace('modele_', ''))
        else:
            modele_charge = donnees_modele
            nom_modele = modele_path.stem.replace('modele_', '')
        
        structured_logger.logger.info(f"✅ Modèle chargé: {nom_modele} - Type: {type(modele_charge)}")
        
        # Test rapide pour vérifier que le preprocesseur fonctionne
        test_data = pd.DataFrame([{
            'bedrooms': 3, 'bathrooms': 2, 'sqft_living': 1800, 'sqft_lot': 7500,
            'floors': 2, 'waterfront': 0, 'view': 0, 'condition': 3,
            'sqft_above': 1800, 'sqft_basement': 0, 'yr_built': 1995
        }])
        
        test_transformed = preprocesseur_charge.transform(test_data)
        structured_logger.logger.info(f"✅ Test preprocesseur réussi - Shape: {test_transformed.shape}")
        
        structured_logger.logger.info("🎯 Système prêt pour les prédictions avec logging complet")
        
        return True
        
    except Exception as e:
        error_msg = f"Erreur chargement: {str(e)}"
        structured_logger.logger.error(f"❌ {error_msg}")
        structured_logger.logger.error(f"Traceback: {traceback.format_exc()}")
        
        structured_logger.log_error("model_loading", error_msg, {
            "traceback": traceback.format_exc(),
            "models_dir": str(models_dir),
            "available_files": [str(p) for p in models_dir.glob("*")] if models_dir.exists() else []
        })
        return False

# Middleware pour mesurer les performances
@app.middleware("http")
async def performance_middleware(request: Request, call_next):
    """Middleware pour mesurer le temps de réponse."""
    start_time = time.time()
    
    # Générer un ID unique pour la requête
    request.state.request_id = str(uuid.uuid4())
    request.state.start_time = start_time
    
    response = await call_next(request)
    
    # Calculer la durée
    duration = (time.time() - start_time) * 1000  # en millisecondes
    response.headers["X-Request-ID"] = request.state.request_id
    response.headers["X-Response-Time"] = f"{duration:.2f}ms"
    
    return response

# Endpoints
@app.get("/health", response_model=HealthResponse)
async def health_check(request: Request):
    """Health check avec logging structuré."""
    start_time = time.time()
    
    try:
        # Vérifier l'état des modèles
        models_ok = modele_charge is not None and preprocesseur_charge is not None
        status = "healthy" if models_ok else "unhealthy"
        
        duration_ms = (time.time() - start_time) * 1000
        
        response = HealthResponse(
            status=status,
            modele_charge=nom_modele if models_ok else "Erreur",
            preprocesseur_charge=preprocesseur_charge is not None,
            timestamp=datetime.utcnow().isoformat() + "Z",
            duration_ms=round(duration_ms, 2),
            environment=os.getenv("MLOPS_ENV", "development")
        )
        
        # Log structuré du health check
        structured_logger.log_health_check(status, models_ok, duration_ms)
        
        if not models_ok:
            raise HTTPException(status_code=500, detail="Modèle ou préprocesseur non chargé")
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        structured_logger.log_error("health_check", str(e), {"duration_ms": duration_ms})
        raise HTTPException(status_code=500, detail="Erreur lors du health check")

@app.post("/predict", response_model=PredictionResponse)
async def predire_prix(features: MaisonFeatures, request: Request):
    """Prédiction avec logging structuré complet."""
    start_time = time.time()
    request_id = getattr(request.state, 'request_id', str(uuid.uuid4()))
    client_ip = request.client.host if request.client else "unknown"
    
    try:
        # Vérifier que les modèles sont chargés
        if modele_charge is None or preprocesseur_charge is None:
            raise HTTPException(status_code=500, detail="Modèle non chargé")
        
        # Convertir en DataFrame
        features_dict = features.dict()
        df_input = pd.DataFrame([features_dict])
        
        structured_logger.logger.info(f"🔧 Input DataFrame shape: {df_input.shape}")
        structured_logger.logger.info(f"🔧 Input columns: {list(df_input.columns)}")
        
        # CORRECTION : Preprocessing avec gestion d'erreur détaillée
        try:
            X_processed = preprocesseur_charge.transform(df_input)
            structured_logger.logger.info(f"✅ Preprocessing réussi - Shape: {X_processed.shape}")
        except Exception as prep_error:
            structured_logger.logger.error(f"❌ Erreur preprocessing: {prep_error}")
            structured_logger.logger.error(f"Preprocesseur type: {type(preprocesseur_charge)}")
            structured_logger.logger.error(f"Preprocesseur attributes: {dir(preprocesseur_charge)}")
            raise Exception(f"Erreur preprocessing: {prep_error}")
        
        # Prédiction
        try:
            prediction_array = modele_charge.predict(X_processed)
            prediction = float(prediction_array[0])
            structured_logger.logger.info(f"✅ Prédiction réussie: {prediction}")
        except Exception as pred_error:
            structured_logger.logger.error(f"❌ Erreur prédiction: {pred_error}")
            raise Exception(f"Erreur prédiction: {pred_error}")
        
        # Calcul de la confiance (exemple simple)
        if hasattr(modele_charge, 'predict_proba'):
            confidence = "Élevée"
        elif prediction > 100000:
            confidence = "Bonne"
        else:
            confidence = "Faible"
        
        # Durée de traitement
        duration_ms = (time.time() - start_time) * 1000
        
        # Réponse structurée
        response = PredictionResponse(
            request_id=request_id,
            prix_predit=round(prediction, 2),
            confiance=confidence,
            modele_utilise=nom_modele,
            timestamp=datetime.utcnow().isoformat() + "Z",
            duration_ms=round(duration_ms, 2)
        )
        
        # LOG STRUCTURÉ COMPLET - EXIGENCE PROJET
        structured_logger.log_prediction(
            request_id=request_id,
            features=features_dict,
            prediction=prediction,
            confidence=confidence,
            model_used=nom_modele,
            duration_ms=duration_ms,
            client_ip=client_ip,
            error=None
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        error_msg = str(e)
        
        structured_logger.logger.error(f"❌ Erreur complète: {error_msg}")
        structured_logger.logger.error(f"Traceback: {traceback.format_exc()}")
        
        # Log de l'erreur
        structured_logger.log_prediction(
            request_id=request_id,
            features=features.dict() if features else {},
            prediction=0.0,
            confidence="Erreur",
            model_used=nom_modele,
            duration_ms=duration_ms,
            client_ip=client_ip,
            error=error_msg
        )
        
        raise HTTPException(status_code=500, detail=f"Erreur lors de la prédiction: {error_msg}")

@app.get("/debug")
async def debug_info():
    """Informations de debug pour le développement."""
    debug_info = {
        "modele_charge": modele_charge is not None,
        "preprocesseur_charge": preprocesseur_charge is not None,
        "nom_modele": nom_modele,
        "environment": os.getenv("MLOPS_ENV", "development"),
        "logging_file": str(structured_logger.log_file),
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }
    
    # Informations détaillées sur les objets chargés
    if modele_charge is not None:
        debug_info["modele_type"] = str(type(modele_charge))
        debug_info["modele_attributes"] = [attr for attr in dir(modele_charge) if not attr.startswith('_')][:10]
    
    if preprocesseur_charge is not None:
        debug_info["preprocesseur_type"] = str(type(preprocesseur_charge))
        debug_info["preprocesseur_attributes"] = [attr for attr in dir(preprocesseur_charge) if not attr.startswith('_')][:10]
    
    return debug_info

@app.get("/logs")
async def get_recent_logs(limit: int = 10):
    """Récupérer les logs récents (développement uniquement)."""
    try:
        if not structured_logger.log_file.exists():
            return {"logs": [], "message": "Aucun log disponible"}
        
        with open(structured_logger.log_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Prendre les dernières lignes
        recent_lines = lines[-limit:] if len(lines) > limit else lines
        
        # Parser le JSON de chaque ligne
        logs = []
        for line in recent_lines:
            try:
                log_entry = json.loads(line.strip())
                logs.append(log_entry)
            except json.JSONDecodeError:
                continue
        
        return {
            "logs": logs,
            "total_logs": len(lines),
            "showing": len(logs)
        }
        
    except Exception as e:
        return {"error": f"Erreur lecture logs: {str(e)}"}

# Event handlers
@app.on_event("startup")
async def startup_event():
    """Initialisation au démarrage."""
    structured_logger.logger.info("🚀 === DÉMARRAGE API AVEC LOGGING STRUCTURÉ ===")
    
    # Charger les modèles
    success = charger_modele_et_preprocesseur()
    
    if success:
        structured_logger.logger.info("✅ API prête avec logging complet")
        structured_logger.logger.info("📊 Logging structuré activé pour audit et monitoring")
    else:
        structured_logger.logger.error("❌ Échec chargement modèles")

@app.on_event("shutdown")
async def shutdown_event():
    """Nettoyage à l'arrêt."""
    structured_logger.logger.info("🛑 Arrêt de l'API MLOps")

# Point d'entrée principal
if __name__ == "__main__":
    print("🚀 === API MLOPS AVEC LOGGING STRUCTURÉ ===")
    print(f"📊 Logs sauvegardés dans: reports/logs/api_audit.log")
    print(f"🔍 Debug disponible sur: /debug")
    print(f"📝 Logs récents sur: /logs")
    
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True if os.getenv("MLOPS_ENV", "development") == "development" else False,
        log_level="info"
    )