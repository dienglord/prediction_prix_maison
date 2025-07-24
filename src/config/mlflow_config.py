import mlflow
import mlflow.sklearn
from pathlib import Path
import os
from datetime import datetime
import logging
import platform

class MLflowConfig:
    """
    Configuration centralisée pour MLflow.
    Compatible Windows et Unix.
    """
    
    def __init__(self, racine_projet: Path = None):
        if racine_projet is None:
            self.racine_projet = Path(__file__).resolve().parents[2]
        else:
            self.racine_projet = Path(racine_projet)
        
        # Configuration des chemins
        self.mlruns_dir = self.racine_projet / "mlruns"
        self.artifacts_dir = self.racine_projet / "mlflow_artifacts"
        
        # Créer les dossiers
        self.mlruns_dir.mkdir(exist_ok=True)
        self.artifacts_dir.mkdir(exist_ok=True)
        
        # Configuration MLflow compatible Windows
        self.experiment_name = "prediction_prix_maison"
        
        # FIX WINDOWS: Utiliser le chemin absolu direct sans file://
        if platform.system() == "Windows":
            self.tracking_uri = str(self.mlruns_dir.resolve())
        else:
            self.tracking_uri = f"file://{self.mlruns_dir}"
        
        self.setup_mlflow()
    
    def setup_mlflow(self):
        """Configure MLflow pour le projet - Version Windows stable."""
        try:
            print(f"🔧 Configuration MLflow...")
            print(f"🖥️ Système: {platform.system()}")
            
            # Configuration ultra-simple pour Windows
            mlruns_dir = self.racine_projet / "mlruns"
            mlruns_dir.mkdir(exist_ok=True)
            
            # URI simple sans file:// pour Windows
            self.tracking_uri = "./mlruns"
            mlflow.set_tracking_uri(self.tracking_uri)
            
            print(f"📁 Tracking URI: {self.tracking_uri}")
            print(f"✅ MLflow configuré")
            
            # CORRECTION: Utiliser une expérience existante ou en créer une
            try:
                # Chercher les expériences existantes
                experiments = mlflow.search_experiments()
                if experiments:
                    # Utiliser la première expérience trouvée
                    exp = experiments[0]
                    mlflow.set_experiment(exp.experiment_id)
                    print(f"✅ Utilisation expérience existante: {exp.name} (ID: {exp.experiment_id})")
                else:
                    # Créer une nouvelle expérience si aucune n'existe
                    exp_id = mlflow.create_experiment(self.experiment_name)
                    mlflow.set_experiment(exp_id)
                    print(f"✅ Nouvelle expérience créée: {self.experiment_name} (ID: {exp_id})")
                    
            except Exception as exp_error:
                print(f"⚠️ Problème expérience: {str(exp_error)}")
                print("🔄 Utilisation mode par défaut...")
            
        except Exception as e:
            print(f"❌ Erreur configuration MLflow: {str(e)}")
            # Configuration minimale en fallback
            self.tracking_uri = "./mlruns"
            print(f"✅ Configuration fallback: {self.tracking_uri}")
    
    def get_run_name(self, model_name: str = None) -> str:
        """Génère un nom de run unique."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if model_name:
            return f"{model_name}_{timestamp}"
        else:
            return f"run_{timestamp}"
    
    def log_system_info(self):
        """Log les informations système."""
        import sys
        
        mlflow.log_param("python_version", sys.version.split()[0])
        mlflow.log_param("platform", platform.platform())
        mlflow.log_param("system", platform.system())
        mlflow.log_param("project_root", str(self.racine_projet))
    
    def log_dataset_info(self, X_train, X_val, X_test, y_train, y_val, y_test):
        """Log les informations sur le dataset."""
        mlflow.log_param("train_samples", len(X_train))
        mlflow.log_param("val_samples", len(X_val))
        mlflow.log_param("test_samples", len(X_test))
        mlflow.log_param("n_features", X_train.shape[1])
        mlflow.log_param("feature_names", list(X_train.columns))
        
        # Statistiques des prix
        mlflow.log_metric("train_price_mean", y_train.mean())
        mlflow.log_metric("train_price_std", y_train.std())
        mlflow.log_metric("val_price_mean", y_val.mean())
        mlflow.log_metric("val_price_std", y_val.std())
        mlflow.log_metric("test_price_mean", y_test.mean())
        mlflow.log_metric("test_price_std", y_test.std())
    
    def start_ui(self, port: int = 5000):
        """Démarre l'interface MLflow UI."""
        print(f"🚀 Démarrage MLflow UI sur le port {port}")
        print(f"📊 URL: http://localhost:{port}")
        print(f"📁 Tracking URI: {self.tracking_uri}")
        
        # Commande pour démarrer l'UI (compatible Windows)
        if platform.system() == "Windows":
            print("\n💡 Pour démarrer l'UI manuellement :")
            print(f"mlflow ui --backend-store-uri \"{self.tracking_uri}\" --port {port}")
        else:
            print("\n💡 Pour démarrer l'UI manuellement :")
            print(f"mlflow ui --backend-store-uri {self.tracking_uri} --port {port}")


def get_mlflow_config() -> MLflowConfig:
    """Factory function pour obtenir la configuration MLflow."""
    return MLflowConfig()