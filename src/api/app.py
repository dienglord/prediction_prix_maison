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
    """Charge le modèle et le preprocesseur avec gestion d'erreurs et extraction automatique."""
    global modele_charge, preprocesseur_charge, nom_modele
    
    try:
        structured_logger.logger.info("🔍 === CHARGEMENT MODÈLE AVEC LOGGING ===")
        
        # Chemin des modèles
        models_dir = Path("models")
        structured_logger.logger.info(f"📊 Dossier models: {models_dir}")
        structured_logger.logger.info(f"📊 Existe: {models_dir.exists()}")
        
        if models_dir.exists():
            files = list(models_dir.glob("*"))
            structured_logger.logger.info(f"📊 Fichiers: {[f.name for f in files]}")
        
        # ===================================================================
        # CHARGEMENT ET CORRECTION AUTOMATIQUE DU PREPROCESSEUR
        # ===================================================================
        preprocesseur_path = models_dir / "preprocesseur.pkl"
        if not preprocesseur_path.exists():
            raise FileNotFoundError(f"Preprocesseur non trouvé: {preprocesseur_path}")
        
        # Charger les données du preprocesseur
        preprocesseur_data = joblib.load(preprocesseur_path)
        structured_logger.logger.info(f"📊 Preprocesseur raw type: {type(preprocesseur_data)}")
        
        # CORRECTION AUTOMATIQUE : Extraire le bon objet
        if isinstance(preprocesseur_data, dict):
            structured_logger.logger.info(f"📊 Preprocesseur est un dict avec clés: {list(preprocesseur_data.keys())}")
            
            # Essayer différentes clés possibles dans l'ordre de priorité
            possible_keys = ['preprocesseur', 'pipeline', 'pipeline_api', 'scaler', 'transformer']
            preprocesseur_found = False
            
            for key in possible_keys:
                if key in preprocesseur_data:
                    candidate = preprocesseur_data[key]
                    if hasattr(candidate, 'transform'):
                        preprocesseur_charge = candidate
                        structured_logger.logger.info(f"✅ Preprocesseur trouvé via clé '{key}': {type(preprocesseur_charge)}")
                        preprocesseur_found = True
                        break
            
            # Si pas trouvé par clé, chercher le premier objet avec transform
            if not preprocesseur_found:
                for key, value in preprocesseur_data.items():
                    if hasattr(value, 'transform') and hasattr(value, 'fit'):
                        preprocesseur_charge = value
                        structured_logger.logger.info(f"✅ Preprocesseur trouvé via scan '{key}': {type(value)}")
                        preprocesseur_found = True
                        break
            
            if not preprocesseur_found:
                # Dernier recours : afficher le contenu pour debug
                structured_logger.logger.error(f"❌ Structure du dict preprocesseur:")
                for key, value in preprocesseur_data.items():
                    structured_logger.logger.error(f"   {key}: {type(value)} - has transform: {hasattr(value, 'transform')}")
                raise ValueError(f"Aucun preprocesseur valide trouvé dans: {list(preprocesseur_data.keys())}")
                
        else:
            # Le preprocesseur est directement l'objet
            if hasattr(preprocesseur_data, 'transform'):
                preprocesseur_charge = preprocesseur_data
                structured_logger.logger.info(f"✅ Preprocesseur direct: {type(preprocesseur_charge)}")
            else:
                raise ValueError(f"L'objet chargé n'a pas de méthode transform: {type(preprocesseur_data)}")
        
        # Test du preprocesseur avec des données réelles
        test_data = pd.DataFrame([{
            'bedrooms': 3, 'bathrooms': 2.0, 'sqft_living': 1800, 'sqft_lot': 7500,
            'floors': 2.0, 'waterfront': 0, 'view': 0, 'condition': 3,
            'sqft_above': 1800, 'sqft_basement': 0, 'yr_built': 1995
        }])
        
        structured_logger.logger.info(f"🧪 Test preprocesseur - Input shape: {test_data.shape}")
        structured_logger.logger.info(f"🧪 Test preprocesseur - Columns: {list(test_data.columns)}")
        
        try:
            test_transformed = preprocesseur_charge.transform(test_data)
            structured_logger.logger.info(f"✅ Test preprocesseur réussi - Output shape: {test_transformed.shape}")
        except Exception as test_error:
            structured_logger.logger.error(f"❌ Test preprocesseur échoué: {test_error}")
            structured_logger.logger.error(f"Preprocesseur methods: {[m for m in dir(preprocesseur_charge) if not m.startswith('_')]}")
            raise Exception(f"Preprocesseur invalide: {test_error}")
        
        # ===================================================================
        # CHARGEMENT DU MODÈLE
        # ===================================================================
        modeles_disponibles = list(models_dir.glob("modele_*.pkl"))
        if not modeles_disponibles:
            raise FileNotFoundError("Aucun modèle trouvé dans models/")
        
        structured_logger.logger.info(f"📊 Modèles disponibles: {[m.name for m in modeles_disponibles]}")
        
        # Charger le premier modèle trouvé (ou implémenter logique de sélection)
        modele_path = modeles_disponibles[0]
        donnees_modele = joblib.load(modele_path)
        
        # Extraire le modèle du dictionnaire si nécessaire
        if isinstance(donnees_modele, dict):
            if 'modele' in donnees_modele:
                modele_charge = donnees_modele['modele']
                nom_modele = donnees_modele.get('nom_modele', modele_path.stem.replace('modele_', ''))
            else:
                # Chercher le premier objet qui a predict
                for key, value in donnees_modele.items():
                    if hasattr(value, 'predict'):
                        modele_charge = value
                        nom_modele = key
                        break
                else:
                    raise ValueError(f"Aucun modèle trouvé dans le dictionnaire: {list(donnees_modele.keys())}")
        else:
            modele_charge = donnees_modele
            nom_modele = modele_path.stem.replace('modele_', '')
        
        structured_logger.logger.info(f"✅ Modèle chargé: {nom_modele} - Type: {type(modele_charge)}")
        
        # Test du modèle avec les données preprocessées
        try:
            test_prediction = modele_charge.predict(test_transformed)
            structured_logger.logger.info(f"✅ Test modèle réussi - Prédiction: {test_prediction[0]:.2f}")
        except Exception as model_test_error:
            structured_logger.logger.error(f"❌ Test modèle échoué: {model_test_error}")
            raise Exception(f"Modèle invalide: {model_test_error}")
        
        structured_logger.logger.info("🎯 === SYSTÈME PRÊT POUR LES PRÉDICTIONS ===")
        structured_logger.logger.info(f"✅ Preprocesseur: {type(preprocesseur_charge).__name__}")
        structured_logger.logger.info(f"✅ Modèle: {nom_modele} ({type(modele_charge).__name__})")
        
        return True
        
    except Exception as e:
        error_msg = f"Erreur chargement: {str(e)}"
        structured_logger.logger.error(f"❌ {error_msg}")
        structured_logger.logger.error(f"Traceback complet: {traceback.format_exc()}")
        
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
    """Prédiction avec logging structuré complet - EXIGENCE PROJET."""
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
        
        structured_logger.logger.info(f"🔧 Prédiction {request_id[:8]} - Input: {df_input.shape}")
        
        # Preprocessing avec gestion d'erreur détaillée
        try:
            X_processed = preprocesseur_charge.transform(df_input)
            structured_logger.logger.info(f"✅ Preprocessing réussi - Shape: {X_processed.shape}")
        except Exception as prep_error:
            structured_logger.logger.error(f"❌ Erreur preprocessing: {prep_error}")
            raise Exception(f"Erreur preprocessing: {prep_error}")
        
        # Prédiction
        try:
            prediction_array = modele_charge.predict(X_processed)
            prediction = float(prediction_array[0])
            structured_logger.logger.info(f"✅ Prédiction réussie: {prediction:.2f}")
        except Exception as pred_error:
            structured_logger.logger.error(f"❌ Erreur prédiction: {pred_error}")
            raise Exception(f"Erreur prédiction: {pred_error}")
        
        # Calcul de la confiance (logique business)
        if hasattr(modele_charge, 'predict_proba'):
            confidence = "Élevée"
        elif prediction > 500000:
            confidence = "Élevée"
        elif prediction > 200000:
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
        
        # ===================================================================
        # LOG STRUCTURÉ COMPLET - EXIGENCE PROJET RESPECTÉE
        # Format: timestamp, features, prédiction, durée, audit
        # ===================================================================
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
        
        structured_logger.logger.error(f"❌ Erreur prédiction complète: {error_msg}")
        
        # Log de l'erreur avec le format structuré
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
    """Informations de debug complètes pour le développement."""
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
        debug_info["modele_methods"] = [m for m in dir(modele_charge) if not m.startswith('_') and callable(getattr(modele_charge, m))][:10]
    
    if preprocesseur_charge is not None:
        debug_info["preprocesseur_type"] = str(type(preprocesseur_charge))
        debug_info["preprocesseur_methods"] = [m for m in dir(preprocesseur_charge) if not m.startswith('_') and callable(getattr(preprocesseur_charge, m))][:10]
    
    return debug_info

@app.get("/logs")
async def get_recent_logs(limit: int = 10):
    """Récupérer les logs récents au format JSON structuré."""
    try:
        if not structured_logger.log_file.exists():
            return {"logs": [], "message": "Aucun log disponible"}
        
        with open(structured_logger.log_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Prendre les dernières lignes et parser le JSON
        recent_lines = lines[-limit:] if len(lines) > limit else lines
        logs = []
        
        for line in recent_lines:
            try:
                log_entry = json.loads(line.strip())
                logs.append(log_entry)
            except json.JSONDecodeError:
                # Ligne non-JSON, ajouter comme texte
                logs.append({"raw_message": line.strip(), "type": "raw"})
        
        return {
            "logs": logs,
            "total_logs": len(lines),
            "showing": len(logs),
            "log_file": str(structured_logger.log_file)
        }
        
    except Exception as e:
        return {"error": f"Erreur lecture logs: {str(e)}"}

@app.get("/metrics")
async def get_metrics():
    """Métriques de base pour monitoring."""
    try:
        # Compter les logs par type
        log_stats = {"predictions": 0, "health_checks": 0, "errors": 0}
        
        if structured_logger.log_file.exists():
            with open(structured_logger.log_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        log_entry = json.loads(line.strip())
                        event_type = log_entry.get("event_type", "unknown")
                        if event_type == "prediction":
                            log_stats["predictions"] += 1
                        elif event_type == "health_check":
                            log_stats["health_checks"] += 1
                        elif event_type == "error":
                            log_stats["errors"] += 1
                    except:
                        continue
        
        return {
            "status": "healthy" if modele_charge and preprocesseur_charge else "unhealthy",
            "model_loaded": nom_modele,
            "uptime_info": {
                "environment": os.getenv("MLOPS_ENV", "development"),
                "api_version": "2.0.0"
            },
            "log_statistics": log_stats,
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
        
    except Exception as e:
        return {"error": f"Erreur métriques: {str(e)}"}

# Event handlers modernes (pas de deprecation warning)
@app.on_event("startup")
async def startup_event():
    """Initialisation au démarrage avec logging structuré."""
    structured_logger.logger.info("🚀 === DÉMARRAGE API MLOPS AVEC LOGGING STRUCTURÉ ===")
    
    # Charger les modèles
    success = charger_modele_et_preprocesseur()
    
    if success:
        structured_logger.logger.info("✅ === API PRÊTE AVEC LOGGING COMPLET ===")
        structured_logger.logger.info("📊 Logging structuré activé pour audit et monitoring")
        structured_logger.logger.info("🎯 Format logs: timestamp, features, prédiction, durée")
    else:
        structured_logger.logger.error("❌ === ÉCHEC CHARGEMENT MODÈLES ===")

@app.on_event("shutdown")
async def shutdown_event():
    """Nettoyage à l'arrêt."""
    structured_logger.logger.info("🛑 === ARRÊT DE L'API MLOPS ===")
    structured_logger.logger.info(f"📊 Logs sauvegardés dans: {structured_logger.log_file}")

# Point d'entrée principal
if __name__ == "__main__":
    print("🚀 === API MLOPS AVEC LOGGING STRUCTURÉ COMPLET ===")
    print(f"📊 Logs sauvegardés dans: reports/logs/api_audit.log")
    print(f"🔍 Debug disponible sur: /debug")
    print(f"📝 Logs récents sur: /logs")
    print(f"📈 Métriques sur: /metrics")
    print("=" * 60)
    
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True if os.getenv("MLOPS_ENV", "development") == "development" else False,
        log_level="info"
    )