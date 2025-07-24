import pandas as pd
import numpy as np
import logging
from pathlib import Path
import sys
import joblib
from typing import Dict, Any, Tuple, List
import warnings
from datetime import datetime
warnings.filterwarnings('ignore')

# MLflow imports
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

# Modèles ML
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration MLflow
sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.config.mlflow_config import get_mlflow_config

class EntraineurMLflow:
    """
    Entraîneur de modèles avec tracking MLflow intégré.
    Phase 2 - MLOps Avancé.
    """
    
    def __init__(self, racine_projet: Path = None):
        if racine_projet is None:
            self.racine_projet = Path(__file__).resolve().parents[2]
        else:
            self.racine_projet = Path(racine_projet)
        
        # Configuration du logging
        self.setup_logging()
        
        # Configuration MLflow
        self.mlflow_config = get_mlflow_config()
        
        # Modèles avec hyperparamètres pour tracking
        self.modeles_config = {
            'linear_regression': {
                'model': LinearRegression(),
                'params': {
                    'fit_intercept': True,
                    'normalize': False
                }
            },
            'random_forest': {
                'model': RandomForestRegressor(
                    n_estimators=100,
                    max_depth=8,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    max_features='sqrt',
                    random_state=42,
                    n_jobs=-1
                ),
                'params': {
                    'n_estimators': 100,
                    'max_depth': 8,
                    'min_samples_split': 5,
                    'min_samples_leaf': 2,
                    'max_features': 'sqrt',
                    'random_state': 42
                }
            },
            'gradient_boosting': {
                'model': GradientBoostingRegressor(
                    n_estimators=50,
                    max_depth=4,
                    learning_rate=0.05,
                    subsample=0.8,
                    max_features='sqrt',
                    random_state=42
                ),
                'params': {
                    'n_estimators': 50,
                    'max_depth': 4,
                    'learning_rate': 0.05,
                    'subsample': 0.8,
                    'max_features': 'sqrt',
                    'random_state': 42
                }
            }
        }
        
        # Stockage des résultats
        self.resultats_entrainement = {}
        self.meilleur_modele = None
        self.meilleur_score = float('-inf')
        self.modele_final = None
        self.meilleur_run_id = None
        
        self.logger.info(f"🔥 Entraîneur MLflow initialisé")
        self.logger.info(f"📊 MLflow tracking: {self.mlflow_config.tracking_uri}")
        self.logger.info(f"🧪 Expérience: {self.mlflow_config.experiment_name}")
    
    def setup_logging(self):
        """Configure le système de logging."""
        logs_dir = self.racine_projet / "reports"
        logs_dir.mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(logs_dir / "training_mlflow.log", encoding='utf-8'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger("entraineur_mlflow")
    
    def charger_donnees_preprocessees(self) -> Tuple[pd.DataFrame, ...]:
        """Charge les données préprocessées."""
        self.logger.info("📊 Chargement des données préprocessées...")
        
        processed_dir = self.racine_projet / "data" / "processed"
        
        try:
            X_train = pd.read_csv(processed_dir / "X_train.csv")
            X_val = pd.read_csv(processed_dir / "X_val.csv")
            X_test = pd.read_csv(processed_dir / "X_test.csv")
            y_train = pd.read_csv(processed_dir / "y_train.csv").iloc[:, 0]
            y_val = pd.read_csv(processed_dir / "y_val.csv").iloc[:, 0]
            y_test = pd.read_csv(processed_dir / "y_test.csv").iloc[:, 0]
            
            self.logger.info(f"✅ Données chargées:")
            self.logger.info(f"   - Train: {X_train.shape}")
            self.logger.info(f"   - Validation: {X_val.shape}")
            self.logger.info(f"   - Test: {X_test.shape}")
            
            return X_train, X_val, X_test, y_train, y_val, y_test
            
        except Exception as e:
            self.logger.error(f"❌ Erreur lors du chargement: {str(e)}")
            raise
    
    def calculer_metriques(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calcule les métriques de performance."""
        metriques = {
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred),
            'mape': np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
        }
        
        return metriques
    
    def entrainer_modele_avec_mlflow(self, nom_modele: str, X_train: pd.DataFrame, y_train: pd.Series,
                                   X_val: pd.DataFrame, y_val: pd.Series) -> Dict[str, Any]:
        """
        Entraîne un modèle avec tracking MLflow - Version Windows stable.
        """
        config = self.modeles_config[nom_modele]
        modele = config['model']
        params = config['params']
        
        # Nom de run simple
        run_name = f"{nom_modele}_{datetime.now().strftime('%H%M%S')}"
        
        try:
            with mlflow.start_run(run_name=run_name) as run:
                self.logger.info(f"🔥 MLflow Run: {run.info.run_id}")
                self.logger.info(f"🔧 Entraînement: {nom_modele}")
                
                # === LOGGING DES PARAMÈTRES ===
                mlflow.log_params(params)
                mlflow.log_param("model_type", nom_modele)
                mlflow.log_param("algorithm", type(modele).__name__)
                
                # === ENTRAÎNEMENT ===
                modele.fit(X_train, y_train)
                
                # === PRÉDICTIONS ===
                y_pred_train = modele.predict(X_train)
                y_pred_val = modele.predict(X_val)
                
                # === MÉTRIQUES ===
                metriques_train = self.calculer_metriques(y_train, y_pred_train)
                metriques_val = self.calculer_metriques(y_val, y_pred_val)
                
                # Validation croisée
                cv_scores = cross_val_score(modele, X_train, y_train, cv=5, 
                                          scoring='neg_mean_squared_error')
                cv_rmse = np.sqrt(-cv_scores)
                
                # Écart train/validation
                ecart_r2 = metriques_train['r2'] - metriques_val['r2']
                
                # === LOGGING DES MÉTRIQUES ===
                mlflow.log_metric("train_rmse", metriques_train['rmse'])
                mlflow.log_metric("train_r2", metriques_train['r2'])
                mlflow.log_metric("val_rmse", metriques_val['rmse'])
                mlflow.log_metric("val_r2", metriques_val['r2'])
                mlflow.log_metric("cv_rmse_mean", cv_rmse.mean())
                mlflow.log_metric("ecart_r2", ecart_r2)
                
                # === LOGGING DU MODÈLE (SIMPLE) ===
                # Pas de model registry pour éviter les erreurs Windows
                mlflow.sklearn.log_model(
                    sk_model=modele,
                    artifact_path="model"
                )
                
                # === TAGS ===
                mlflow.set_tag("model_family", "regression")
                mlflow.set_tag("use_case", "prix_immobilier")
                
                # === LOGGING CONSOLE ===
                self.logger.info(f"✅ {nom_modele} entraîné:")
                self.logger.info(f"   - R² validation: {metriques_val['r2']:.4f}")
                self.logger.info(f"   - RMSE validation: {metriques_val['rmse']:.2f}")
                self.logger.info(f"   - MLflow Run ID: {run.info.run_id}")
                
                # Préparer les résultats
                resultats = {
                    'modele': modele,
                    'nom': nom_modele,
                    'run_id': run.info.run_id,
                    'metriques_train': metriques_train,
                    'metriques_val': metriques_val,
                    'cv_rmse_mean': cv_rmse.mean(),
                    'ecart_r2': ecart_r2,
                    'y_pred_train': y_pred_train,
                    'y_pred_val': y_pred_val
                }
                
                return resultats
                
        except Exception as e:
            self.logger.warning(f"⚠️ MLflow indisponible pour {nom_modele}: {str(e)}")
            self.logger.info(f"🔄 Entraînement sans MLflow...")
            
            # Fallback sans MLflow - fonctionne parfaitement
            modele.fit(X_train, y_train)
            y_pred_train = modele.predict(X_train)
            y_pred_val = modele.predict(X_val)
            
            metriques_train = self.calculer_metriques(y_train, y_pred_train)
            metriques_val = self.calculer_metriques(y_val, y_pred_val)
            
            cv_scores = cross_val_score(modele, X_train, y_train, cv=5, 
                                      scoring='neg_mean_squared_error')
            cv_rmse = np.sqrt(-cv_scores)
            ecart_r2 = metriques_train['r2'] - metriques_val['r2']
            
            self.logger.info(f"✅ {nom_modele} (sans MLflow):")
            self.logger.info(f"   - R² validation: {metriques_val['r2']:.4f}")
            self.logger.info(f"   - RMSE validation: {metriques_val['rmse']:.2f}")
            
            return {
                'modele': modele,
                'nom': nom_modele,
                'run_id': 'no_mlflow',
                'metriques_train': metriques_train,
                'metriques_val': metriques_val,
                'cv_rmse_mean': cv_rmse.mean(),
                'ecart_r2': ecart_r2,
                'y_pred_train': y_pred_train,
                'y_pred_val': y_pred_val
            }
    
    def creer_visualisations_mlflow(self, y_train, y_pred_train, y_val, y_pred_val, 
                                   nom_modele, run_id):
        """Crée et sauvegarde des visualisations pour MLflow."""
        
        # Créer dossier temporaire dans le projet
        temp_dir = self.racine_projet / "temp_plots"
        temp_dir.mkdir(exist_ok=True)
        
        # 1. Prédictions vs Vraies valeurs
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Training set
        ax1.scatter(y_train, y_pred_train, alpha=0.5, color='blue')
        ax1.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2)
        ax1.set_xlabel('Prix réels')
        ax1.set_ylabel('Prix prédits')
        ax1.set_title(f'{nom_modele} - Training Set')
        ax1.grid(True, alpha=0.3)
        
        # Validation set
        ax2.scatter(y_val, y_pred_val, alpha=0.5, color='orange')
        ax2.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--', lw=2)
        ax2.set_xlabel('Prix réels')
        ax2.set_ylabel('Prix prédits')
        ax2.set_title(f'{nom_modele} - Validation Set')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Sauvegarder dans le projet et logger
        temp_path = temp_dir / f"predictions_{nom_modele}_{run_id}.png"
        plt.savefig(temp_path, dpi=300, bbox_inches='tight')
        mlflow.log_artifact(str(temp_path), "plots")
        plt.close()
        
        # 2. Résidus
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Résidus training
        residus_train = y_train - y_pred_train
        ax1.scatter(y_pred_train, residus_train, alpha=0.5, color='blue')
        ax1.axhline(y=0, color='r', linestyle='--')
        ax1.set_xlabel('Prix prédits')
        ax1.set_ylabel('Résidus')
        ax1.set_title(f'{nom_modele} - Résidus Training')
        ax1.grid(True, alpha=0.3)
        
        # Résidus validation
        residus_val = y_val - y_pred_val
        ax2.scatter(y_pred_val, residus_val, alpha=0.5, color='orange')
        ax2.axhline(y=0, color='r', linestyle='--')
        ax2.set_xlabel('Prix prédits')
        ax2.set_ylabel('Résidus')
        ax2.set_title(f'{nom_modele} - Résidus Validation')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Sauvegarder et logger
        temp_path_residus = temp_dir / f"residus_{nom_modele}_{run_id}.png"
        plt.savefig(temp_path_residus, dpi=300, bbox_inches='tight')
        mlflow.log_artifact(str(temp_path_residus), "plots")
        plt.close()
        
        # Nettoyer les fichiers temporaires après logging
        try:
            temp_path.unlink()
            temp_path_residus.unlink()
        except:
            pass
    
    def entrainer_tous_modeles_mlflow(self, X_train: pd.DataFrame, y_train: pd.Series,
                                     X_val: pd.DataFrame, y_val: pd.Series) -> Dict[str, Any]:
        """
        Entraîne tous les modèles avec MLflow tracking - Version Windows stable.
        """
        self.logger.info("🔥 === ENTRAÎNEMENT AVEC MLFLOW ===")
        
        self.resultats_entrainement = {}
        
        for nom_modele in self.modeles_config.keys():
            try:
                resultats = self.entrainer_modele_avec_mlflow(
                    nom_modele, X_train, y_train, X_val, y_val
                )
                self.resultats_entrainement[nom_modele] = resultats
                
                # Suivre le meilleur modèle
                r2_val = resultats['metriques_val']['r2']
                if r2_val > self.meilleur_score:
                    self.meilleur_score = r2_val
                    self.meilleur_modele = nom_modele
                    self.modele_final = resultats['modele']
                    self.meilleur_run_id = resultats['run_id']
                
            except Exception as e:
                self.logger.error(f"❌ Échec {nom_modele}: {str(e)}")
                continue
        
        self.logger.info(f"🏆 Meilleur modèle: {self.meilleur_modele} (R² = {self.meilleur_score:.4f})")
        
        return self.resultats_entrainement
    
    def evaluer_sur_test_mlflow(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """Évalue le meilleur modèle sur test - avec fallback."""
        if self.modele_final is None:
            raise ValueError("Aucun modèle n'a été entraîné")
        
        # Prédictions sur le test
        y_pred_test = self.modele_final.predict(X_test)
        metriques_test = self.calculer_metriques(y_test, y_pred_test)
        
        # Essayer de logger dans MLflow
        try:
            with mlflow.start_run(run_name=f"test_evaluation_{self.meilleur_modele}"):
                mlflow.log_param("model_name", self.meilleur_modele)
                mlflow.log_param("best_run_id", self.meilleur_run_id)
                
                mlflow.log_metric("test_rmse", metriques_test['rmse'])
                mlflow.log_metric("test_mae", metriques_test['mae'])
                mlflow.log_metric("test_r2", metriques_test['r2'])
                
                # Écart validation/test
                r2_val = self.resultats_entrainement[self.meilleur_modele]['metriques_val']['r2']
                ecart_val_test = r2_val - metriques_test['r2']
                mlflow.log_metric("generalization_gap", ecart_val_test)
                
                mlflow.set_tag("run_type", "test_evaluation")
                mlflow.set_tag("final_model", "true")
                
                self.logger.info(f"✅ Évaluation test loggée dans MLflow")
                
        except Exception as e:
            self.logger.warning(f"⚠️ MLflow indisponible pour test: {str(e)}")
            self.logger.info(f"📊 Évaluation test sans MLflow...")
        
        # Calcul de l'écart validation/test
        r2_val = self.resultats_entrainement[self.meilleur_modele]['metriques_val']['r2']
        ecart_val_test = r2_val - metriques_test['r2']
        
        self.logger.info(f"🎯 Évaluation test - {self.meilleur_modele}:")
        self.logger.info(f"   - R² test: {metriques_test['r2']:.4f}")
        self.logger.info(f"   - RMSE test: {metriques_test['rmse']:.2f}")
        self.logger.info(f"   - Écart val/test: {ecart_val_test:.4f}")
        
        return metriques_test


def main():
    """Pipeline d'entraînement avec MLflow."""
    try:
        print("🔥 === ENTRAÎNEMENT AVEC MLFLOW - PHASE 2 ===")
        
        # Initialiser l'entraîneur MLflow
        entraineur = EntraineurMLflow()
        
        print(f"\n📊 MLflow UI: mlflow ui --backend-store-uri {entraineur.mlflow_config.tracking_uri}")
        
        print("\n📊 1. Chargement des données...")
        X_train, X_val, X_test, y_train, y_val, y_test = entraineur.charger_donnees_preprocessees()
        
        print("\n🔥 2. Entraînement avec MLflow tracking...")
        resultats = entraineur.entrainer_tous_modeles_mlflow(X_train, y_train, X_val, y_val)
        
        print("\n🎯 3. Évaluation finale sur test...")
        metriques_test = entraineur.evaluer_sur_test_mlflow(X_test, y_test)
        
        print("\n✅ === ENTRAÎNEMENT MLFLOW TERMINÉ ===")
        print(f"🏆 Meilleur modèle: {entraineur.meilleur_modele}")
        print(f"📊 R² test: {metriques_test['r2']:.4f}")
        print(f"🔥 Run ID: {entraineur.meilleur_run_id}")
        print(f"\n🎯 Pour voir les résultats:")
        print(f"mlflow ui --backend-store-uri {entraineur.mlflow_config.tracking_uri}")
        print("Puis allez sur: http://localhost:5000")
        
        return entraineur
        
    except Exception as e:
        print(f"❌ Erreur: {str(e)}")
        return None


if __name__ == "__main__":
    entraineur = main()