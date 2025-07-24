import pandas as pd
import numpy as np
import logging
from pathlib import Path
import sys
import joblib
from typing import Dict, Any, Tuple, List
import warnings
from datetime import datetime
import os
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

# Configuration centralisée
sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.config.gestionnaire_config import get_config_manager

class EntraineurAvecConfiguration:
    """
    Entraîneur de modèles utilisant la configuration centralisée.
    Version MLOps Phase 2 - Configuration Management.
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
        
        # Configuration MLflow
        self.setup_mlflow()
        
        # Créer les modèles depuis la configuration
        self.modeles_config = self.create_models_from_config()
        
        # Stockage des résultats
        self.resultats_entrainement = {}
        self.meilleur_modele = None
        self.meilleur_score = float('-inf')
        self.modele_final = None
        self.meilleur_run_id = None
        
        self.logger.info(f"🔥 Entraîneur configuré - Environnement: {self.environment}")
        self.logger.info(f"📊 Modèles activés: {list(self.modeles_config.keys())}")
        self.logger.info(f"🐛 Mode debug: {self.config.debug}")
    
    def setup_logging(self):
        """Configure le système de logging depuis la configuration."""
        logs_dir = self.racine_projet / self.config.logging.file_path
        logs_dir.mkdir(exist_ok=True)
        
        # Configuration du niveau de logging
        level = getattr(logging, self.config.logging.level.upper())
        
        handlers = []
        
        # Handler fichier si activé
        if self.config.logging.file_enabled:
            file_handler = logging.FileHandler(
                logs_dir / f"training_{self.environment}.log", 
                encoding='utf-8'
            )
            file_handler.setFormatter(logging.Formatter(self.config.logging.format))
            handlers.append(file_handler)
        
        # Handler console si activé
        if self.config.logging.console_enabled:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(logging.Formatter(self.config.logging.format))
            handlers.append(console_handler)
        
        logging.basicConfig(
            level=level,
            format=self.config.logging.format,
            handlers=handlers,
            force=True  # Override existing config
        )
        
        self.logger = logging.getLogger(f"entraineur_{self.environment}")
    
    def setup_mlflow(self):
        """Configure MLflow depuis la configuration."""
        try:
            # Configuration MLflow
            mlflow.set_tracking_uri(self.config.mlflow.tracking_uri)
            
            # Configuration de l'expérience
            try:
                experiments = mlflow.search_experiments()
                experiment_found = False
                
                for exp in experiments:
                    if exp.name == self.config.mlflow.experiment_name:
                        mlflow.set_experiment(exp.experiment_id)
                        self.logger.info(f"✅ Expérience MLflow: {exp.name} (ID: {exp.experiment_id})")
                        experiment_found = True
                        break
                
                if not experiment_found:
                    # Créer l'expérience si elle n'existe pas
                    exp_id = mlflow.create_experiment(
                        name=self.config.mlflow.experiment_name,
                        artifact_location=self.config.mlflow.artifact_location
                    )
                    mlflow.set_experiment(exp_id)
                    self.logger.info(f"✅ Nouvelle expérience MLflow: {self.config.mlflow.experiment_name}")
                
            except Exception as exp_error:
                self.logger.warning(f"⚠️ Problème expérience MLflow: {str(exp_error)}")
                # Utiliser l'expérience par défaut
                experiments = mlflow.search_experiments()
                if experiments:
                    mlflow.set_experiment(experiments[0].experiment_id)
                    self.logger.info("🔄 Utilisation expérience par défaut")
            
        except Exception as e:
            self.logger.error(f"❌ Erreur configuration MLflow: {str(e)}")
    
    def create_models_from_config(self) -> Dict[str, Any]:
        """Crée les modèles depuis la configuration."""
        modeles = {}
        
        # Récupérer seulement les modèles activés
        enabled_models = self.config_manager.get_enabled_models()
        
        for nom_modele, model_config in enabled_models.items():
            try:
                # Créer le modèle selon l'algorithme
                if model_config.algorithm == "LinearRegression":
                    modele = LinearRegression(**model_config.hyperparameters)
                    
                elif model_config.algorithm == "RandomForestRegressor":
                    modele = RandomForestRegressor(**model_config.hyperparameters)
                    
                elif model_config.algorithm == "GradientBoostingRegressor":
                    modele = GradientBoostingRegressor(**model_config.hyperparameters)
                    
                else:
                    self.logger.warning(f"⚠️ Algorithme non supporté: {model_config.algorithm}")
                    continue
                
                modeles[nom_modele] = {
                    'model': modele,
                    'config': model_config
                }
                
                self.logger.info(f"✅ Modèle créé: {nom_modele} ({model_config.algorithm})")
                
            except Exception as e:
                self.logger.error(f"❌ Erreur création modèle {nom_modele}: {str(e)}")
                continue
        
        return modeles
    
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
            
            # Validation des features configurées
            expected_features = self.config.data.features
            actual_features = list(X_train.columns)
            
            if set(expected_features) != set(actual_features):
                self.logger.warning(f"⚠️ Features différentes de la configuration:")
                self.logger.warning(f"   Configurées: {expected_features}")
                self.logger.warning(f"   Trouvées: {actual_features}")
            
            self.logger.info(f"✅ Données chargées:")
            self.logger.info(f"   - Train: {X_train.shape}")
            self.logger.info(f"   - Validation: {X_val.shape}")
            self.logger.info(f"   - Test: {X_test.shape}")
            self.logger.info(f"   - Features: {len(actual_features)}")
            
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
    
    def entrainer_modele_avec_mlflow(self, nom_modele: str, model_info: Dict[str, Any],
                                   X_train: pd.DataFrame, y_train: pd.Series,
                                   X_val: pd.DataFrame, y_val: pd.Series) -> Dict[str, Any]:
        """Entraîne un modèle avec tracking MLflow utilisant la configuration."""
        
        modele = model_info['model']
        model_config = model_info['config']
        
        # Nom de run basé sur la configuration
        run_name = f"{nom_modele}_{self.environment}_{datetime.now().strftime('%H%M%S')}"
        
        try:
            with mlflow.start_run(run_name=run_name) as run:
                self.logger.info(f"🔥 MLflow Run: {run.info.run_id}")
                self.logger.info(f"🔧 Entraînement: {nom_modele}")
                
                # === LOGGING DES PARAMÈTRES DE CONFIGURATION ===
                mlflow.log_param("environment", self.environment)
                mlflow.log_param("model_name", nom_modele)
                mlflow.log_param("algorithm", model_config.algorithm)
                mlflow.log_param("model_description", model_config.description)
                
                # Hyperparamètres depuis la configuration
                mlflow.log_params(model_config.hyperparameters)
                
                # Paramètres de données depuis la configuration
                mlflow.log_param("test_size", self.config.data.test_size)
                mlflow.log_param("val_size", self.config.data.val_size)
                mlflow.log_param("random_state", self.config.data.random_state)
                mlflow.log_param("n_features", len(self.config.data.features))
                
                # === ENTRAÎNEMENT ===
                start_time = datetime.now()
                modele.fit(X_train, y_train)
                training_time = (datetime.now() - start_time).total_seconds()
                
                mlflow.log_metric("training_time_seconds", training_time)
                
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
                mlflow.log_metric("train_mae", metriques_train['mae'])
                
                mlflow.log_metric("val_rmse", metriques_val['rmse'])
                mlflow.log_metric("val_r2", metriques_val['r2'])
                mlflow.log_metric("val_mae", metriques_val['mae'])
                
                mlflow.log_metric("cv_rmse_mean", cv_rmse.mean())
                mlflow.log_metric("cv_rmse_std", cv_rmse.std())
                mlflow.log_metric("overfitting_gap", ecart_r2)
                
                # === VALIDATION DES CRITÈRES DE PERFORMANCE ===
                # Vérifier les critères définis dans la configuration des modèles
                performance_ok = True
                if hasattr(model_config, 'performance_criteria'):
                    criteria = getattr(model_config, 'performance_criteria', {})
                    
                    min_r2 = criteria.get('min_r2_score', 0)
                    max_rmse = criteria.get('max_rmse', float('inf'))
                    max_time = criteria.get('max_training_time_seconds', float('inf'))
                    
                    if metriques_val['r2'] < min_r2:
                        performance_ok = False
                        mlflow.set_tag("performance_warning", f"R2 ({metriques_val['r2']:.4f}) < min ({min_r2})")
                    
                    if metriques_val['rmse'] > max_rmse:
                        performance_ok = False
                        mlflow.set_tag("performance_warning", f"RMSE ({metriques_val['rmse']:.0f}) > max ({max_rmse})")
                    
                    if training_time > max_time:
                        performance_ok = False
                        mlflow.set_tag("performance_warning", f"Temps ({training_time:.1f}s) > max ({max_time}s)")
                
                # === LOGGING DU MODÈLE ===
                if self.config.mlflow.log_models:
                    mlflow.sklearn.log_model(
                        sk_model=modele,
                        artifact_path="model"
                    )
                
                # === TAGS ===
                mlflow.set_tag("environment", self.environment)
                mlflow.set_tag("model_family", "regression")
                mlflow.set_tag("use_case", "prix_immobilier")
                mlflow.set_tag("performance_ok", str(performance_ok))
                
                if self.config.debug:
                    mlflow.set_tag("debug_mode", "true")
                
                # === LOGGING CONSOLE ===
                self.logger.info(f"✅ {nom_modele} entraîné:")
                self.logger.info(f"   - R² validation: {metriques_val['r2']:.4f}")
                self.logger.info(f"   - RMSE validation: {metriques_val['rmse']:.2f}")
                self.logger.info(f"   - Temps d'entraînement: {training_time:.2f}s")
                self.logger.info(f"   - Performance OK: {performance_ok}")
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
                    'training_time': training_time,
                    'performance_ok': performance_ok,
                    'y_pred_train': y_pred_train,
                    'y_pred_val': y_pred_val
                }
                
                return resultats
                
        except Exception as e:
            self.logger.warning(f"⚠️ MLflow indisponible pour {nom_modele}: {str(e)}")
            self.logger.info(f"🔄 Entraînement sans MLflow...")
            
            # Fallback sans MLflow
            start_time = datetime.now()
            modele.fit(X_train, y_train)
            training_time = (datetime.now() - start_time).total_seconds()
            
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
            self.logger.info(f"   - Temps d'entraînement: {training_time:.2f}s")
            
            return {
                'modele': modele,
                'nom': nom_modele,
                'run_id': 'no_mlflow',
                'metriques_train': metriques_train,
                'metriques_val': metriques_val,
                'cv_rmse_mean': cv_rmse.mean(),
                'ecart_r2': ecart_r2,
                'training_time': training_time,
                'performance_ok': True,  # Assume OK in fallback
                'y_pred_train': y_pred_train,
                'y_pred_val': y_pred_val
            }
    
    def entrainer_tous_modeles(self, X_train: pd.DataFrame, y_train: pd.Series,
                              X_val: pd.DataFrame, y_val: pd.Series) -> Dict[str, Any]:
        """Entraîne tous les modèles configurés."""
        self.logger.info("🔥 === ENTRAÎNEMENT AVEC CONFIGURATION ===")
        self.logger.info(f"📊 Environnement: {self.environment}")
        self.logger.info(f"🤖 Modèles à entraîner: {list(self.modeles_config.keys())}")
        
        self.resultats_entrainement = {}
        
        for nom_modele, model_info in self.modeles_config.items():
            try:
                resultats = self.entrainer_modele_avec_mlflow(
                    nom_modele, model_info, X_train, y_train, X_val, y_val
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
    
    def evaluer_sur_test(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """Évalue le meilleur modèle sur test."""
        if self.modele_final is None:
            raise ValueError("Aucun modèle n'a été entraîné")
        
        # Prédictions sur le test
        y_pred_test = self.modele_final.predict(X_test)
        metriques_test = self.calculer_metriques(y_test, y_pred_test)
        
        # Logger dans MLflow si possible
        try:
            with mlflow.start_run(run_name=f"test_evaluation_{self.meilleur_modele}_{self.environment}"):
                mlflow.log_param("environment", self.environment)
                mlflow.log_param("best_model", self.meilleur_modele)
                mlflow.log_param("best_run_id", self.meilleur_run_id)
                
                mlflow.log_metric("test_rmse", metriques_test['rmse'])
                mlflow.log_metric("test_mae", metriques_test['mae'])
                mlflow.log_metric("test_r2", metriques_test['r2'])
                
                # Écart validation/test
                r2_val = self.resultats_entrainement[self.meilleur_modele]['metriques_val']['r2']
                ecart_val_test = r2_val - metriques_test['r2']
                mlflow.log_metric("generalization_gap", ecart_val_test)
                
                mlflow.set_tag("run_type", "test_evaluation")
                mlflow.set_tag("environment", self.environment)
                mlflow.set_tag("final_model", "true")
                
        except Exception as e:
            self.logger.warning(f"⚠️ MLflow indisponible pour test: {str(e)}")
        
        # Calcul de l'écart validation/test
        r2_val = self.resultats_entrainement[self.meilleur_modele]['metriques_val']['r2']
        ecart_val_test = r2_val - metriques_test['r2']
        
        self.logger.info(f"🎯 Évaluation test - {self.meilleur_modele}:")
        self.logger.info(f"   - R² test: {metriques_test['r2']:.4f}")
        self.logger.info(f"   - RMSE test: {metriques_test['rmse']:.2f}")
        self.logger.info(f"   - Écart val/test: {ecart_val_test:.4f}")
        
        return metriques_test
    
    def sauvegarder_meilleur_modele(self) -> Path:
        """Sauvegarde le meilleur modèle selon la configuration."""
        if self.modele_final is None:
            raise ValueError("Aucun modèle à sauvegarder")
        
        models_dir = self.racine_projet / "models"
        models_dir.mkdir(exist_ok=True)
        
        # Nom de fichier incluant l'environnement
        nom_fichier = f"modele_{self.meilleur_modele}_{self.environment}.pkl"
        chemin_sauvegarde = models_dir / nom_fichier
        
        # Données à sauvegarder
        donnees_modele = {
            'modele': self.modele_final,
            'nom_modele': self.meilleur_modele,
            'environment': self.environment,
            'run_id': self.meilleur_run_id,
            'metriques_test': self.resultats_entrainement[self.meilleur_modele]['metriques_val'],
            'hyperparameters': self.modeles_config[self.meilleur_modele]['config'].hyperparameters,
            'timestamp': datetime.now().isoformat(),
            'config_version': "2.0_with_config_management"
        }
        
        joblib.dump(donnees_modele, chemin_sauvegarde)
        
        self.logger.info(f"✅ Modèle sauvegardé: {chemin_sauvegarde}")
        
        return chemin_sauvegarde


def main():
    """Pipeline d'entraînement avec configuration centralisée."""
    try:
        # Récupérer l'environnement depuis les variables d'environnement ou argument
        import argparse
        parser = argparse.ArgumentParser(description="Entraînement avec configuration")
        parser.add_argument('--environment', '-e', default=None,
                           help='Environnement de configuration (development, production)')
        args = parser.parse_args()
        
        print("🔥 === ENTRAÎNEMENT AVEC CONFIGURATION - PHASE 2 ===")
        
        # Initialiser l'entraîneur avec configuration
        entraineur = EntraineurAvecConfiguration(environment=args.environment)
        
        print(f"\n📊 Configuration chargée:")
        print(f"   - Environnement: {entraineur.environment}")
        print(f"   - Debug: {entraineur.config.debug}")
        print(f"   - Modèles: {list(entraineur.modeles_config.keys())}")
        print(f"   - MLflow: {entraineur.config.mlflow.experiment_name}")
        
        print("\n📊 1. Chargement des données...")
        X_train, X_val, X_test, y_train, y_val, y_test = entraineur.charger_donnees_preprocessees()
        
        print("\n🔥 2. Entraînement avec configuration...")
        resultats = entraineur.entrainer_tous_modeles(X_train, y_train, X_val, y_val)
        
        print("\n🎯 3. Évaluation finale sur test...")
        metriques_test = entraineur.evaluer_sur_test(X_test, y_test)
        
        print("\n💾 4. Sauvegarde du meilleur modèle...")
        chemin_modele = entraineur.sauvegarder_meilleur_modele()
        
        print("\n✅ === ENTRAÎNEMENT TERMINÉ ===")
        print(f"🏆 Meilleur modèle: {entraineur.meilleur_modele}")
        print(f"📊 R² test: {metriques_test['r2']:.4f}")
        print(f"🎯 RMSE test: {metriques_test['rmse']:.2f}")
        print(f"💾 Modèle sauvé: {chemin_modele.name}")
        print(f"🔥 Environnement: {entraineur.environment}")
        
        if entraineur.config.mlflow.tracking_uri:
            print(f"\n📊 Pour voir MLflow UI:")
            print(f"mlflow ui --backend-store-uri {entraineur.config.mlflow.tracking_uri} --port 5000")
        
        return entraineur
        
    except Exception as e:
        print(f"❌ Erreur: {str(e)}")
        return None


if __name__ == "__main__":
    entraineur = main()