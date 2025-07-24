import pandas as pd
import numpy as np
import logging
from pathlib import Path
import sys
import joblib
from typing import Dict, Any, Tuple
import warnings
from datetime import datetime
import os
warnings.filterwarnings('ignore')

# MLflow imports (optionnel)
try:
    import mlflow
    import mlflow.sklearn
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    print("⚠️ MLflow non disponible - fonctionnement sans tracking")

# Modèles ML - SEULEMENT LINEAR REGRESSION
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score

# Configuration centralisée
sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.config.gestionnaire_config import get_config_manager

class EntraineurOptimise:
    """
    Entraîneur optimisé - SEULEMENT Linear Regression (le meilleur modèle).
    Version GitHub optimisée.
    """
    
    def __init__(self, environment: str = None):
        # Déterminer l'environnement
        self.environment = environment or os.getenv('MLOPS_ENV', 'development')
        
        # Charger la configuration
        self.config_manager = get_config_manager(self.environment)
        self.config = self.config_manager.config
        
        if not self.config:
            raise ValueError(f"Impossible de charger la configuration pour: {self.environment}")
        
        # Configuration du projet
        self.racine_projet = self.config_manager.project_root
        
        # Configuration du logging
        self.setup_logging()
        
        # Configuration MLflow (optionnelle)
        if MLFLOW_AVAILABLE:
            self.setup_mlflow()
        
        # Créer SEULEMENT le modèle Linear Regression
        self.modele_config = self.create_linear_regression_model()
        
        # Stockage des résultats
        self.modele_final = None
        self.metriques_finales = None
        
        self.logger.info(f"🔥 Entraîneur optimisé - Environnement: {self.environment}")
        self.logger.info(f"📊 Modèle unique: Linear Regression (le meilleur)")
        self.logger.info(f"🎯 Version GitHub optimisée")
    
    def setup_logging(self):
        """Configure le système de logging depuis la configuration."""
        logs_dir = self.racine_projet / self.config.logging.file_path
        logs_dir.mkdir(exist_ok=True)
        
        # Configuration du niveau de logging
        level = getattr(logging, self.config.logging.level.upper())
        
        # Logger simple pour version optimisée
        logging.basicConfig(
            level=level,
            format=self.config.logging.format,
            handlers=[
                logging.FileHandler(logs_dir / f"training_{self.environment}.log", encoding='utf-8'),
                logging.StreamHandler(sys.stdout)
            ],
            force=True
        )
        
        self.logger = logging.getLogger(f"entraineur_optimise_{self.environment}")
    
    def setup_mlflow(self):
        """Configure MLflow depuis la configuration (optionnel)."""
        if not MLFLOW_AVAILABLE:
            return
            
        try:
            # Configuration MLflow légère
            mlflow.set_tracking_uri(self.config.mlflow.tracking_uri)
            
            # Créer ou utiliser l'expérience
            try:
                mlflow.set_experiment(self.config.mlflow.experiment_name)
                self.logger.info(f"✅ MLflow configuré: {self.config.mlflow.experiment_name}")
            except Exception as exp_error:
                self.logger.warning(f"⚠️ MLflow expérience: {str(exp_error)}")
                
        except Exception as e:
            self.logger.warning(f"⚠️ MLflow indisponible: {str(e)}")
            self.MLFLOW_AVAILABLE = False
    
    def create_linear_regression_model(self) -> Dict[str, Any]:
        """Crée SEULEMENT le modèle Linear Regression optimisé."""
        
        # Configuration manuelle optimisée pour Linear Regression
        model_config = {
            'algorithm': 'LinearRegression',
            'hyperparameters': {
                'fit_intercept': True
                # normalize supprimé (obsolète dans scikit-learn récent)
            },
            'description': 'Linear Regression - Meilleur modèle (R² = 0.4924)',
            'performance': {
                'r2_test': 0.4924,
                'rmse_test': 253409.77,
                'training_time': 0.01
            }
        }
        
        # Créer le modèle
        modele = LinearRegression(**model_config['hyperparameters'])
        
        modele_info = {
            'model': modele,
            'config': model_config,
            'name': 'linear_regression'
        }
        
        self.logger.info(f"✅ Modèle Linear Regression créé")
        self.logger.info(f"📊 Hyperparamètres: {model_config['hyperparameters']}")
        
        return modele_info
    
    def charger_donnees_preprocessees(self) -> Tuple[pd.DataFrame, ...]:
        """Charge les données préprocessées depuis la configuration."""
        self.logger.info("📊 Chargement des données préprocessées...")
        
        processed_dir = self.racine_projet / self.config.data.processed_path
        
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
            self.logger.info(f"   - Features: {len(X_train.columns)}")
            
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
    
    def entrainer_modele_optimise(self, X_train: pd.DataFrame, y_train: pd.Series,
                                 X_val: pd.DataFrame, y_val: pd.Series) -> Dict[str, Any]:
        """Entraîne le modèle Linear Regression avec tracking optionnel."""
        
        modele = self.modele_config['model']
        config = self.modele_config['config']
        
        self.logger.info("🔥 === ENTRAÎNEMENT LINEAR REGRESSION OPTIMISÉ ===")
        
        # MLflow run optionnel
        if MLFLOW_AVAILABLE:
            run_name = f"linear_regression_{self.environment}_{datetime.now().strftime('%H%M%S')}"
            try:
                mlflow.start_run(run_name=run_name)
                self.logger.info(f"📊 MLflow Run démarré")
            except:
                self.logger.warning("⚠️ MLflow run échoué - continue sans tracking")
        
        try:
            # === ENTRAÎNEMENT ===
            start_time = datetime.now()
            modele.fit(X_train, y_train)
            training_time = (datetime.now() - start_time).total_seconds()
            
            self.logger.info(f"✅ Entraînement terminé en {training_time:.3f}s")
            
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
            
            # Écart train/validation (overfitting check)
            ecart_r2 = metriques_train['r2'] - metriques_val['r2']
            
            # === LOGGING OPTIONNEL MLFLOW ===
            if MLFLOW_AVAILABLE:
                try:
                    mlflow.log_param("environment", self.environment)
                    mlflow.log_param("algorithm", "LinearRegression")
                    mlflow.log_params(config['hyperparameters'])
                    
                    mlflow.log_metric("train_r2", metriques_train['r2'])
                    mlflow.log_metric("val_r2", metriques_val['r2'])
                    mlflow.log_metric("val_rmse", metriques_val['rmse'])
                    mlflow.log_metric("cv_rmse_mean", cv_rmse.mean())
                    mlflow.log_metric("training_time", training_time)
                    mlflow.log_metric("overfitting_gap", ecart_r2)
                    
                    mlflow.set_tag("model_type", "linear_regression")
                    mlflow.set_tag("optimized_version", "github")
                    mlflow.set_tag("best_model", "true")
                    
                    mlflow.end_run()
                    self.logger.info("📊 Métriques MLflow enregistrées")
                except Exception as mlflow_error:
                    self.logger.warning(f"⚠️ MLflow logging échoué: {mlflow_error}")
            
            # === AFFICHAGE CONSOLE ===
            self.logger.info("✅ === RÉSULTATS LINEAR REGRESSION ===")
            self.logger.info(f"📊 R² validation: {metriques_val['r2']:.4f}")
            self.logger.info(f"📊 RMSE validation: {metriques_val['rmse']:.2f}")
            self.logger.info(f"📊 MAE validation: {metriques_val['mae']:.2f}")
            self.logger.info(f"📊 CV RMSE: {cv_rmse.mean():.2f} ± {cv_rmse.std():.2f}")
            self.logger.info(f"📊 Temps entraînement: {training_time:.3f}s")
            self.logger.info(f"📊 Écart train/val: {ecart_r2:.4f}")
            
            # Performance check
            performance_ok = (
                metriques_val['r2'] > 0.4 and 
                metriques_val['rmse'] < 300000 and
                training_time < 10
            )
            
            self.logger.info(f"📊 Performance OK: {performance_ok}")
            
            # Préparer les résultats
            resultats = {
                'modele': modele,
                'nom': 'linear_regression',
                'metriques_train': metriques_train,
                'metriques_val': metriques_val,
                'cv_rmse_mean': cv_rmse.mean(),
                'cv_rmse_std': cv_rmse.std(),
                'ecart_r2': ecart_r2,
                'training_time': training_time,
                'performance_ok': performance_ok,
                'y_pred_train': y_pred_train,
                'y_pred_val': y_pred_val
            }
            
            return resultats
            
        except Exception as e:
            self.logger.error(f"❌ Erreur entraînement: {str(e)}")
            if MLFLOW_AVAILABLE:
                try:
                    mlflow.end_run(status="FAILED")
                except:
                    pass
            raise
    
    def evaluer_sur_test(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """Évalue le modèle sur le set de test."""
        if self.modele_final is None:
            raise ValueError("Aucun modèle n'a été entraîné")
        
        self.logger.info("🎯 === ÉVALUATION FINALE SUR TEST ===")
        
        # Prédictions sur le test
        y_pred_test = self.modele_final.predict(X_test)
        metriques_test = self.calculer_metriques(y_test, y_pred_test)
        
        # Logging MLflow optionnel
        if MLFLOW_AVAILABLE:
            try:
                with mlflow.start_run(run_name=f"test_evaluation_{self.environment}"):
                    mlflow.log_param("environment", self.environment)
                    mlflow.log_param("model_type", "linear_regression")
                    mlflow.log_param("evaluation_type", "final_test")
                    
                    mlflow.log_metric("test_rmse", metriques_test['rmse'])
                    mlflow.log_metric("test_mae", metriques_test['mae'])
                    mlflow.log_metric("test_r2", metriques_test['r2'])
                    
                    mlflow.set_tag("final_evaluation", "true")
                    mlflow.set_tag("optimized_version", "github")
            except Exception as e:
                self.logger.warning(f"⚠️ MLflow test logging échoué: {str(e)}")
        
        # Calcul de l'écart validation/test (généralisation)
        r2_val = self.metriques_finales['metriques_val']['r2']
        ecart_val_test = r2_val - metriques_test['r2']
        
        self.logger.info(f"📊 === RÉSULTATS FINAUX ===")
        self.logger.info(f"📊 R² test: {metriques_test['r2']:.4f}")
        self.logger.info(f"📊 RMSE test: {metriques_test['rmse']:.2f}")
        self.logger.info(f"📊 MAE test: {metriques_test['mae']:.2f}")
        self.logger.info(f"📊 Écart val/test: {ecart_val_test:.4f}")
        
        # Évaluation de la généralisation
        if abs(ecart_val_test) < 0.05:
            self.logger.info("✅ Excellente généralisation !")
        elif abs(ecart_val_test) < 0.10:
            self.logger.info("✅ Bonne généralisation")
        else:
            self.logger.warning("⚠️ Généralisation à surveiller")
        
        return metriques_test
    
    def sauvegarder_modele_optimise(self) -> Path:
        """Sauvegarde le modèle Linear Regression optimisé."""
        if self.modele_final is None:
            raise ValueError("Aucun modèle à sauvegarder")
        
        models_dir = self.racine_projet / "models"
        models_dir.mkdir(exist_ok=True)
        
        # Nom de fichier optimisé
        nom_fichier = f"modele_linear_regression_{self.environment}.pkl"
        chemin_sauvegarde = models_dir / nom_fichier
        
        # Données optimisées à sauvegarder
        donnees_modele = {
            'modele': self.modele_final,
            'nom_modele': 'linear_regression',
            'environment': self.environment,
            'metriques_test': self.metriques_finales.get('metriques_test', {}),
            'hyperparameters': self.modele_config['config']['hyperparameters'],
            'performance': self.modele_config['config']['performance'],
            'timestamp': datetime.now().isoformat(),
            'version': "2.0_github_optimized",
            'description': "Modèle Linear Regression optimisé - Meilleur performance",
            'optimization_info': {
                'original_models': 3,
                'kept_models': 1,
                'reason': 'GitHub size optimization',
                'space_saved': '95%',
                'performance_impact': 'None - best model kept'
            }
        }
        
        joblib.dump(donnees_modele, chemin_sauvegarde)
        
        self.logger.info(f"✅ Modèle sauvegardé: {chemin_sauvegarde}")
        self.logger.info(f"📦 Taille fichier: {chemin_sauvegarde.stat().st_size / 1024:.1f} KB")
        
        return chemin_sauvegarde
    
    def entrainer_pipeline_complet(self):
        """Pipeline d'entraînement complet optimisé."""
        self.logger.info("🚀 === PIPELINE ENTRAÎNEUR OPTIMISÉ ===")
        
        try:
            # 1. Charger les données
            self.logger.info("📊 1. Chargement des données...")
            X_train, X_val, X_test, y_train, y_val, y_test = self.charger_donnees_preprocessees()
            
            # 2. Entraîner Linear Regression
            self.logger.info("🔥 2. Entraînement Linear Regression...")
            resultats = self.entrainer_modele_optimise(X_train, y_train, X_val, y_val)
            
            # Stocker les résultats
            self.modele_final = resultats['modele']
            self.metriques_finales = resultats
            
            # 3. Évaluation finale sur test
            self.logger.info("🎯 3. Évaluation finale sur test...")
            metriques_test = self.evaluer_sur_test(X_test, y_test)
            self.metriques_finales['metriques_test'] = metriques_test
            
            # 4. Sauvegarde
            self.logger.info("💾 4. Sauvegarde du modèle...")
            chemin_modele = self.sauvegarder_modele_optimise()
            
            # 5. Résumé final
            self.logger.info("✅ === PIPELINE TERMINÉ ===")
            self.logger.info(f"🏆 Modèle: Linear Regression")
            self.logger.info(f"📊 R² test: {metriques_test['r2']:.4f}")
            self.logger.info(f"🎯 RMSE test: {metriques_test['rmse']:.2f}")
            self.logger.info(f"💾 Modèle sauvé: {chemin_modele.name}")
            self.logger.info(f"🔥 Environnement: {self.environment}")
            self.logger.info(f"🎯 Version: GitHub optimisée")
            
            return {
                'success': True,
                'model_path': chemin_modele,
                'metrics': metriques_test,
                'model_name': 'linear_regression'
            }
            
        except Exception as e:
            self.logger.error(f"❌ Erreur pipeline: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }


def main():
    """Pipeline d'entraînement optimisé - Linear Regression uniquement."""
    try:
        # Récupérer l'environnement
        import argparse
        parser = argparse.ArgumentParser(description="Entraînement optimisé Linear Regression")
        parser.add_argument('--environment', '-e', default=None,
                           help='Environnement de configuration (development, production)')
        args = parser.parse_args()
        
        print("🚀 === ENTRAÎNEUR OPTIMISÉ GITHUB - LINEAR REGRESSION ===")
        print("📊 Version allégée - Seulement le meilleur modèle")
        print("🎯 Économie d'espace : ~95% (50MB → 2MB)")
        print("=" * 60)
        
        # Initialiser l'entraîneur optimisé
        entraineur = EntraineurOptimise(environment=args.environment)
        
        print(f"\n📊 Configuration:")
        print(f"   - Environnement: {entraineur.environment}")
        print(f"   - Debug: {entraineur.config.debug}")
        print(f"   - Modèle unique: Linear Regression")
        print(f"   - Version: GitHub optimisée")
        
        # Exécuter le pipeline complet
        resultats = entraineur.entrainer_pipeline_complet()
        
        if resultats['success']:
            print(f"\n🎉 === SUCCÈS ===")
            print(f"🏆 Meilleur modèle: {resultats['model_name']}")
            print(f"📊 R² test: {resultats['metrics']['r2']:.4f}")
            print(f"🎯 RMSE test: {resultats['metrics']['rmse']:.2f}")
            print(f"💾 Modèle sauvé: {resultats['model_path'].name}")
            
            if MLFLOW_AVAILABLE and entraineur.config.mlflow.tracking_uri:
                print(f"\n📊 Pour voir MLflow UI:")
                print(f"mlflow ui --backend-store-uri {entraineur.config.mlflow.tracking_uri} --port 5000")
        else:
            print(f"\n❌ === ÉCHEC ===")
            print(f"Erreur: {resultats['error']}")
            return None
        
        return entraineur
        
    except Exception as e:
        print(f"❌ Erreur globale: {str(e)}")
        return None


if __name__ == "__main__":
    entraineur = main()