import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional, Union
import logging
from dataclasses import dataclass, field
from datetime import datetime

@dataclass
class ModelConfig:
    """Configuration d'un modèle ML."""
    name: str
    algorithm: str
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    description: str = ""

@dataclass
class DataConfig:
    """Configuration des données."""
    raw_path: str = "data/raw/data.csv"
    processed_path: str = "data/processed"
    test_size: float = 0.2
    val_size: float = 0.1
    random_state: int = 42
    features: list = field(default_factory=lambda: [
        'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
        'waterfront', 'view', 'condition', 'sqft_above', 'sqft_basement', 'yr_built'
    ])

@dataclass 
class APIConfig:
    """Configuration de l'API."""
    host: str = "0.0.0.0"
    port: int = 8000
    title: str = "API Prédiction Prix Maisons"
    description: str = "API MLOps pour prédire le prix des maisons"
    version: str = "1.0.0"
    reload: bool = False
    log_level: str = "info"

@dataclass
class MLflowConfig:
    """Configuration MLflow."""
    tracking_uri: str = "./mlruns"
    experiment_name: str = "prediction_prix_maison"
    artifact_location: str = "./mlflow_artifacts"
    auto_log: bool = True
    log_models: bool = True

@dataclass
class LoggingConfig:
    """Configuration du logging."""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_enabled: bool = True
    file_path: str = "reports/logs"
    console_enabled: bool = True

@dataclass
class EnvironmentConfig:
    """Configuration globale pour un environnement."""
    name: str
    debug: bool = False
    data: DataConfig = field(default_factory=DataConfig)
    api: APIConfig = field(default_factory=APIConfig)
    mlflow: MLflowConfig = field(default_factory=MLflowConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    models: Dict[str, ModelConfig] = field(default_factory=dict)

class ConfigurationManager:
    """
    Gestionnaire centralisé des configurations MLOps.
    
    Gère le chargement, la validation et l'accès aux configurations
    pour tous les composants du système.
    """
    
    def __init__(self, project_root: Optional[Path] = None, environment: str = "development"):
        self.project_root = project_root or Path(__file__).resolve().parents[2]
        self.config_dir = self.project_root / "config"
        self.environment = environment
        
        # Configuration chargée
        self.config: Optional[EnvironmentConfig] = None
        
        # Logger
        self.logger = self._setup_logger()
        
        # Charger la configuration
        self.load_configuration()
    
    def _setup_logger(self) -> logging.Logger:
        """Configure le logger pour le gestionnaire de config."""
        logger = logging.getLogger("config_manager")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def load_configuration(self) -> None:
        """Charge la configuration pour l'environnement spécifié."""
        try:
            self.logger.info(f"🔧 Chargement configuration environnement: {self.environment}")
            
            # Créer la structure de base si elle n'existe pas
            self._ensure_config_structure()
            
            # Charger la configuration d'environnement
            env_config = self._load_environment_config()
            
            # Charger les configurations des modèles
            models_config = self._load_models_config()
            
            # Combiner les configurations
            self.config = EnvironmentConfig(
                name=self.environment,
                debug=env_config.get('debug', False),
                data=self._create_data_config(env_config.get('data', {})),
                api=self._create_api_config(env_config.get('api', {})),
                mlflow=self._create_mlflow_config(env_config.get('mlflow', {})),
                logging=self._create_logging_config(env_config.get('logging', {})),
                models=models_config
            )
            
            # Valider la configuration
            self._validate_configuration()
            
            self.logger.info(f"✅ Configuration chargée avec succès")
            self.logger.info(f"   - Environnement: {self.environment}")
            self.logger.info(f"   - Modèles: {list(self.config.models.keys())}")
            self.logger.info(f"   - Debug: {self.config.debug}")
            
        except Exception as e:
            self.logger.error(f"❌ Erreur chargement configuration: {str(e)}")
            # Configuration par défaut en fallback
            self.config = EnvironmentConfig(name=self.environment)
            self.logger.warning("⚠️ Utilisation configuration par défaut")
    
    def _ensure_config_structure(self) -> None:
        """Crée la structure de configuration si elle n'existe pas."""
        
        # Créer les dossiers
        self.config_dir.mkdir(exist_ok=True)
        (self.config_dir / "environments").mkdir(exist_ok=True)
        (self.config_dir / "models").mkdir(exist_ok=True)
        
        # Créer les fichiers par défaut s'ils n'existent pas
        self._create_default_configs()
    
    def _create_default_configs(self) -> None:
        """Crée les fichiers de configuration par défaut."""
        
        # Configuration d'environnement development par défaut
        dev_config_path = self.config_dir / "environments" / "development.yaml"
        if not dev_config_path.exists():
            default_dev_config = {
                'debug': True,
                'api': {
                    'port': 8000,
                    'reload': True,
                    'log_level': 'debug'
                },
                'logging': {
                    'level': 'DEBUG',
                    'console_enabled': True,
                    'file_enabled': True
                },
                'mlflow': {
                    'auto_log': True,
                    'log_models': True
                }
            }
            with open(dev_config_path, 'w') as f:
                yaml.dump(default_dev_config, f, default_flow_style=False)
        
        # Configuration des modèles par défaut
        models = {
            'linear_regression': {
                'algorithm': 'LinearRegression',
                'hyperparameters': {
                    'fit_intercept': True,
                    'normalize': False
                },
                'enabled': True,
                'description': 'Régression linéaire simple et rapide'
            },
            'random_forest': {
                'algorithm': 'RandomForestRegressor',
                'hyperparameters': {
                    'n_estimators': 100,
                    'max_depth': 8,
                    'min_samples_split': 5,
                    'min_samples_leaf': 2,
                    'max_features': 'sqrt',
                    'random_state': 42,
                    'n_jobs': -1
                },
                'enabled': True,
                'description': 'Random Forest optimisé pour éviter le surapprentissage'
            },
            'gradient_boosting': {
                'algorithm': 'GradientBoostingRegressor',
                'hyperparameters': {
                    'n_estimators': 50,
                    'max_depth': 4,
                    'learning_rate': 0.05,
                    'subsample': 0.8,
                    'max_features': 'sqrt',
                    'random_state': 42
                },
                'enabled': True,
                'description': 'Gradient Boosting avec régularisation'
            }
        }
        
        for model_name, model_config in models.items():
            model_path = self.config_dir / "models" / f"{model_name}.yaml"
            if not model_path.exists():
                with open(model_path, 'w') as f:
                    yaml.dump(model_config, f, default_flow_style=False)
    
    def _load_environment_config(self) -> Dict[str, Any]:
        """Charge la configuration d'environnement."""
        env_path = self.config_dir / "environments" / f"{self.environment}.yaml"
        
        if env_path.exists():
            with open(env_path, 'r') as f:
                return yaml.safe_load(f) or {}
        else:
            self.logger.warning(f"⚠️ Fichier d'environnement non trouvé: {env_path}")
            return {}
    
    def _load_models_config(self) -> Dict[str, ModelConfig]:
        """Charge les configurations des modèles."""
        models_config = {}
        models_dir = self.config_dir / "models"
        
        if models_dir.exists():
            for model_file in models_dir.glob("*.yaml"):
                try:
                    with open(model_file, 'r') as f:
                        model_data = yaml.safe_load(f)
                    
                    model_name = model_file.stem
                    models_config[model_name] = ModelConfig(
                        name=model_name,
                        algorithm=model_data.get('algorithm', ''),
                        hyperparameters=model_data.get('hyperparameters', {}),
                        enabled=model_data.get('enabled', True),
                        description=model_data.get('description', '')
                    )
                    
                except Exception as e:
                    self.logger.warning(f"⚠️ Erreur chargement modèle {model_file}: {e}")
        
        return models_config
    
    def _create_data_config(self, data_dict: Dict[str, Any]) -> DataConfig:
        """Crée la configuration des données."""
        return DataConfig(
            raw_path=data_dict.get('raw_path', 'data/raw/data.csv'),
            processed_path=data_dict.get('processed_path', 'data/processed'),
            test_size=data_dict.get('test_size', 0.2),
            val_size=data_dict.get('val_size', 0.1),
            random_state=data_dict.get('random_state', 42),
            features=data_dict.get('features', DataConfig().features)
        )
    
    def _create_api_config(self, api_dict: Dict[str, Any]) -> APIConfig:
        """Crée la configuration de l'API."""
        return APIConfig(
            host=api_dict.get('host', '0.0.0.0'),
            port=api_dict.get('port', 8000),
            title=api_dict.get('title', 'API Prédiction Prix Maisons'),
            description=api_dict.get('description', 'API MLOps pour prédire le prix des maisons'),
            version=api_dict.get('version', '1.0.0'),
            reload=api_dict.get('reload', False),
            log_level=api_dict.get('log_level', 'info')
        )
    
    def _create_mlflow_config(self, mlflow_dict: Dict[str, Any]) -> MLflowConfig:
        """Crée la configuration MLflow."""
        return MLflowConfig(
            tracking_uri=mlflow_dict.get('tracking_uri', './mlruns'),
            experiment_name=mlflow_dict.get('experiment_name', 'prediction_prix_maison'),
            artifact_location=mlflow_dict.get('artifact_location', './mlflow_artifacts'),
            auto_log=mlflow_dict.get('auto_log', True),
            log_models=mlflow_dict.get('log_models', True)
        )
    
    def _create_logging_config(self, logging_dict: Dict[str, Any]) -> LoggingConfig:
        """Crée la configuration du logging."""
        return LoggingConfig(
            level=logging_dict.get('level', 'INFO'),
            format=logging_dict.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s'),
            file_enabled=logging_dict.get('file_enabled', True),
            file_path=logging_dict.get('file_path', 'reports/logs'),
            console_enabled=logging_dict.get('console_enabled', True)
        )
    
    def _validate_configuration(self) -> None:
        """Valide la configuration chargée."""
        if not self.config:
            raise ValueError("Configuration non chargée")
        
        # Validation des modèles
        if not self.config.models:
            self.logger.warning("⚠️ Aucun modèle configuré")
        
        # Validation des chemins
        data_path = self.project_root / self.config.data.raw_path
        if not data_path.parent.exists():
            self.logger.warning(f"⚠️ Dossier de données manquant: {data_path.parent}")
        
        # Validation des ports
        if not (1000 <= self.config.api.port <= 65535):
            self.logger.warning(f"⚠️ Port API invalide: {self.config.api.port}")
    
    def get_model_config(self, model_name: str) -> Optional[ModelConfig]:
        """Retourne la configuration d'un modèle spécifique."""
        if not self.config:
            return None
        return self.config.models.get(model_name)
    
    def get_enabled_models(self) -> Dict[str, ModelConfig]:
        """Retourne les modèles activés."""
        if not self.config:
            return {}
        return {name: config for name, config in self.config.models.items() if config.enabled}
    
    def override_config(self, **kwargs) -> None:
        """Permet de surcharger des paramètres de configuration."""
        if not self.config:
            return
        
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
                self.logger.info(f"🔧 Override: {key} = {value}")
    
    def save_config(self, path: Optional[Path] = None) -> None:
        """Sauvegarde la configuration actuelle."""
        if not self.config:
            return
        
        if path is None:
            path = self.config_dir / f"current_config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.yaml"
        
        # Convertir en dictionnaire pour sauvegarde
        config_dict = {
            'environment': self.config.name,
            'debug': self.config.debug,
            'data': self.config.data.__dict__,
            'api': self.config.api.__dict__,
            'mlflow': self.config.mlflow.__dict__,
            'logging': self.config.logging.__dict__,
            'models': {name: {
                'algorithm': model.algorithm,
                'hyperparameters': model.hyperparameters,
                'enabled': model.enabled,
                'description': model.description
            } for name, model in self.config.models.items()}
        }
        
        with open(path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
        
        self.logger.info(f"💾 Configuration sauvegardée: {path}")
    
    def reload_configuration(self) -> None:
        """Recharge la configuration depuis les fichiers."""
        self.load_configuration()
    
    def set_environment(self, environment: str) -> None:
        """Change l'environnement et recharge la configuration."""
        self.environment = environment
        self.reload_configuration()


# Factory function pour faciliter l'utilisation
def get_config_manager(environment: str = None, project_root: Path = None) -> ConfigurationManager:
    """
    Factory function pour obtenir un gestionnaire de configuration.
    
    Args:
        environment: Environnement à charger (development, staging, production)
        project_root: Racine du projet
    
    Returns:
        ConfigurationManager: Instance configurée
    """
    if environment is None:
        environment = os.getenv('MLOPS_ENV', 'development')
    
    return ConfigurationManager(project_root=project_root, environment=environment)


# Instance globale pour faciliter les imports
_config_manager = None

def get_config() -> ConfigurationManager:
    """Retourne l'instance globale du gestionnaire de configuration."""
    global _config_manager
    if _config_manager is None:
        _config_manager = get_config_manager()
    return _config_manager