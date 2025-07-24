#!/usr/bin/env python3
"""
CLI pour la gestion des configurations MLOps.
Permet de valider, visualiser et manipuler les configurations.
"""

import argparse
import sys
import json
from pathlib import Path
from typing import Dict, Any
import yaml

# Ajouter le projet au path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config.gestionnaire_config import get_config_manager, ConfigurationManager

def validate_config(args):
    """Valide une configuration."""
    print(f"🔍 Validation de la configuration: {args.environment}")
    
    try:
        config_manager = get_config_manager(args.environment)
        
        if config_manager.config is None:
            print("❌ Configuration non chargée")
            return False
        
        print("✅ Configuration valide")
        print(f"   - Environnement: {config_manager.config.name}")
        print(f"   - Debug: {config_manager.config.debug}")
        print(f"   - Modèles activés: {len(config_manager.get_enabled_models())}")
        
        # Validation des modèles
        enabled_models = config_manager.get_enabled_models()
        for name, model_config in enabled_models.items():
            print(f"   - ✅ {name}: {model_config.algorithm}")
        
        disabled_models = {name: model for name, model in config_manager.config.models.items() 
                          if not model.enabled}
        for name in disabled_models:
            print(f"   - ⚠️ {name}: Désactivé")
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur validation: {str(e)}")
        return False

def show_config(args):
    """Affiche la configuration courante."""
    print(f"📊 Configuration: {args.environment}")
    
    try:
        config_manager = get_config_manager(args.environment)
        
        if config_manager.config is None:
            print("❌ Configuration non disponible")
            return
        
        config = config_manager.config
        
        print(f"\n🌍 Environnement: {config.name}")
        print(f"🐛 Debug: {config.debug}")
        
        print(f"\n📊 Données:")
        print(f"   - Fichier brut: {config.data.raw_path}")
        print(f"   - Dossier traité: {config.data.processed_path}")
        print(f"   - Taille test: {config.data.test_size}")
        print(f"   - Taille validation: {config.data.val_size}")
        print(f"   - Features: {len(config.data.features)}")
        
        print(f"\n🚀 API:")
        print(f"   - Host: {config.api.host}")
        print(f"   - Port: {config.api.port}")
        print(f"   - Version: {config.api.version}")
        print(f"   - Reload: {config.api.reload}")
        print(f"   - Log level: {config.api.log_level}")
        
        print(f"\n📈 MLflow:")
        print(f"   - Tracking URI: {config.mlflow.tracking_uri}")
        print(f"   - Expérience: {config.mlflow.experiment_name}")
        print(f"   - Auto-log: {config.mlflow.auto_log}")
        
        print(f"\n🤖 Modèles ({len(config.models)}):")
        for name, model in config.models.items():
            status = "✅" if model.enabled else "❌"
            print(f"   {status} {name} ({model.algorithm})")
            if args.verbose:
                print(f"      Description: {model.description}")
                print(f"      Hyperparamètres: {len(model.hyperparameters)}")
        
        if args.verbose:
            print(f"\n🔧 Hyperparamètres détaillés:")
            for name, model in config.models.items():
                if model.enabled:
                    print(f"\n   {name}:")
                    for param, value in model.hyperparameters.items():
                        print(f"     - {param}: {value}")
        
    except Exception as e:
        print(f"❌ Erreur affichage: {str(e)}")

def list_environments(args):
    """Liste les environnements disponibles."""
    print("🌍 Environnements disponibles:")
    
    try:
        config_dir = Path("config/environments")
        if not config_dir.exists():
            print("❌ Dossier d'environnements non trouvé")
            return
        
        env_files = list(config_dir.glob("*.yaml"))
        if not env_files:
            print("❌ Aucun environnement configuré")
            return
        
        for env_file in sorted(env_files):
            env_name = env_file.stem
            try:
                # Charger la config pour vérifier qu'elle est valide
                config_manager = get_config_manager(env_name)
                status = "✅" if config_manager.config else "❌"
                debug_status = "🐛" if config_manager.config and config_manager.config.debug else "🚀"
                
                print(f"   {status} {debug_status} {env_name}")
                
                if args.verbose and config_manager.config:
                    print(f"      Port API: {config_manager.config.api.port}")
                    print(f"      Modèles: {len(config_manager.config.models)}")
                    
            except Exception as e:
                print(f"   ❌ {env_name} (erreur: {str(e)})")
        
    except Exception as e:
        print(f"❌ Erreur listage: {str(e)}")

def list_models(args):
    """Liste les modèles disponibles."""
    print("🤖 Modèles disponibles:")
    
    try:
        config_dir = Path("config/models")
        if not config_dir.exists():
            print("❌ Dossier de modèles non trouvé")
            return
        
        model_files = list(config_dir.glob("*.yaml"))
        if not model_files:
            print("❌ Aucun modèle configuré")
            return
        
        for model_file in sorted(model_files):
            model_name = model_file.stem
            try:
                with open(model_file, 'r') as f:
                    model_data = yaml.safe_load(f)
                
                enabled = model_data.get('enabled', True)
                algorithm = model_data.get('algorithm', 'Inconnu')
                description = model_data.get('description', 'Pas de description')
                
                status = "✅" if enabled else "❌"
                print(f"   {status} {model_name} ({algorithm})")
                
                if args.verbose:
                    print(f"      Description: {description}")
                    hyperparams = model_data.get('hyperparameters', {})
                    print(f"      Hyperparamètres: {len(hyperparams)}")
                    
                    if hyperparams:
                        for param, value in list(hyperparams.items())[:3]:  # Limite à 3
                            print(f"        - {param}: {value}")
                        if len(hyperparams) > 3:
                            print(f"        ... et {len(hyperparams) - 3} autres")
                    
            except Exception as e:
                print(f"   ❌ {model_name} (erreur: {str(e)})")
        
    except Exception as e:
        print(f"❌ Erreur listage modèles: {str(e)}")

def export_config(args):
    """Exporte la configuration vers un fichier."""
    print(f"💾 Export configuration: {args.environment}")
    
    try:
        config_manager = get_config_manager(args.environment)
        
        if config_manager.config is None:
            print("❌ Configuration non disponible")
            return
        
        # Créer le dictionnaire d'export
        export_data = {
            'environment': config_manager.config.name,
            'exported_at': str(config_manager.config.__class__.__module__),
            'debug': config_manager.config.debug,
            'data': {
                'raw_path': config_manager.config.data.raw_path,
                'processed_path': config_manager.config.data.processed_path,
                'test_size': config_manager.config.data.test_size,
                'val_size': config_manager.config.data.val_size,
                'random_state': config_manager.config.data.random_state,
                'features': config_manager.config.data.features
            },
            'api': {
                'host': config_manager.config.api.host,
                'port': config_manager.config.api.port,
                'title': config_manager.config.api.title,
                'version': config_manager.config.api.version,
                'reload': config_manager.config.api.reload,
                'log_level': config_manager.config.api.log_level
            },
            'mlflow': {
                'tracking_uri': config_manager.config.mlflow.tracking_uri,
                'experiment_name': config_manager.config.mlflow.experiment_name,
                'auto_log': config_manager.config.mlflow.auto_log,
                'log_models': config_manager.config.mlflow.log_models
            },
            'models': {}
        }
        
        # Ajouter les modèles
        for name, model in config_manager.config.models.items():
            export_data['models'][name] = {
                'algorithm': model.algorithm,
                'enabled': model.enabled,
                'description': model.description,
                'hyperparameters': model.hyperparameters
            }
        
        # Déterminer le format de sortie
        output_file = Path(args.output) if args.output else Path(f"config_export_{args.environment}.json")
        
        if output_file.suffix.lower() == '.yaml':
            with open(output_file, 'w') as f:
                yaml.dump(export_data, f, default_flow_style=False, indent=2)
        else:
            with open(output_file, 'w') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Configuration exportée: {output_file}")
        print(f"   - Format: {'YAML' if output_file.suffix.lower() == '.yaml' else 'JSON'}")
        print(f"   - Taille: {output_file.stat().st_size} bytes")
        
    except Exception as e:
        print(f"❌ Erreur export: {str(e)}")

def check_differences(args):
    """Compare deux environnements."""
    print(f"🔍 Comparaison: {args.env1} vs {args.env2}")
    
    try:
        config1 = get_config_manager(args.env1)
        config2 = get_config_manager(args.env2)
        
        if not config1.config or not config2.config:
            print("❌ Impossible de charger les configurations")
            return
        
        differences = []
        
        # Comparer les configurations de base
        if config1.config.debug != config2.config.debug:
            differences.append(f"Debug: {config1.config.debug} vs {config2.config.debug}")
        
        if config1.config.api.port != config2.config.api.port:
            differences.append(f"Port API: {config1.config.api.port} vs {config2.config.api.port}")
        
        if config1.config.api.reload != config2.config.api.reload:
            differences.append(f"API Reload: {config1.config.api.reload} vs {config2.config.api.reload}")
        
        if config1.config.mlflow.experiment_name != config2.config.mlflow.experiment_name:
            differences.append(f"Expérience MLflow: {config1.config.mlflow.experiment_name} vs {config2.config.mlflow.experiment_name}")
        
        # Comparer les modèles
        models1 = set(config1.config.models.keys())
        models2 = set(config2.config.models.keys())
        
        only_in_1 = models1 - models2
        only_in_2 = models2 - models1
        common_models = models1 & models2
        
        if only_in_1:
            differences.append(f"Modèles uniquement dans {args.env1}: {', '.join(only_in_1)}")
        
        if only_in_2:
            differences.append(f"Modèles uniquement dans {args.env2}: {', '.join(only_in_2)}")
        
        # Comparer les modèles communs
        for model_name in common_models:
            model1 = config1.config.models[model_name]
            model2 = config2.config.models[model_name]
            
            if model1.enabled != model2.enabled:
                differences.append(f"Modèle {model_name} activé: {model1.enabled} vs {model2.enabled}")
            
            if model1.hyperparameters != model2.hyperparameters:
                differences.append(f"Hyperparamètres {model_name}: Différents")
        
        if differences:
            print(f"⚠️ {len(differences)} différence(s) trouvée(s):")
            for i, diff in enumerate(differences, 1):
                print(f"   {i}. {diff}")
        else:
            print("✅ Aucune différence majeure trouvée")
        
    except Exception as e:
        print(f"❌ Erreur comparaison: {str(e)}")

def init_config(args):
    """Initialise la structure de configuration."""
    print("🏗️ Initialisation de la structure de configuration...")
    
    try:
        # Créer un gestionnaire pour forcer la création des fichiers
        config_manager = get_config_manager(args.environment)
        
        print("✅ Structure de configuration initialisée")
        print(f"   - Environnement: {args.environment}")
        print(f"   - Dossier config: {config_manager.config_dir}")
        
        # Lister les fichiers créés
        env_files = list((config_manager.config_dir / "environments").glob("*.yaml"))
        model_files = list((config_manager.config_dir / "models").glob("*.yaml"))
        
        print(f"   - Environnements: {len(env_files)}")
        print(f"   - Modèles: {len(model_files)}")
        
        if args.verbose:
            print("\n📁 Fichiers créés:")
            for f in env_files + model_files:
                print(f"   - {f.relative_to(config_manager.project_root)}")
        
    except Exception as e:
        print(f"❌ Erreur initialisation: {str(e)}")

def main():
    """Point d'entrée principal du CLI."""
    parser = argparse.ArgumentParser(
        description="🔧 CLI de gestion des configurations MLOps",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples d'utilisation:
  python config_cli.py validate --environment development
  python config_cli.py show --environment production --verbose
  python config_cli.py list-models --verbose
  python config_cli.py export --environment development --output config.yaml
  python config_cli.py diff --env1 development --env2 production
        """
    )
    
    parser.add_argument('--verbose', '-v', action='store_true', 
                       help='Affichage détaillé')
    
    subparsers = parser.add_subparsers(dest='command', help='Commandes disponibles')
    
    # Commande validate
    validate_parser = subparsers.add_parser('validate', help='Valider une configuration')
    validate_parser.add_argument('--environment', '-e', default='development',
                                help='Environnement à valider')
    validate_parser.set_defaults(func=validate_config)
    
    # Commande show
    show_parser = subparsers.add_parser('show', help='Afficher une configuration')
    show_parser.add_argument('--environment', '-e', default='development',
                            help='Environnement à afficher')
    show_parser.add_argument('--verbose', '-v', action='store_true', 
                           help='Affichage détaillé')
    show_parser.set_defaults(func=show_config)
    
    # Commande list-environments
    list_env_parser = subparsers.add_parser('list-environments', help='Lister les environnements')
    list_env_parser.add_argument('--verbose', '-v', action='store_true', 
                                help='Affichage détaillé')
    list_env_parser.set_defaults(func=list_environments)
    
    # Commande list-models
    list_models_parser = subparsers.add_parser('list-models', help='Lister les modèles')
    list_models_parser.add_argument('--verbose', '-v', action='store_true', 
                                   help='Affichage détaillé')
    list_models_parser.set_defaults(func=list_models)
    
    # Commande export
    export_parser = subparsers.add_parser('export', help='Exporter une configuration')
    export_parser.add_argument('--environment', '-e', default='development',
                              help='Environnement à exporter')
    export_parser.add_argument('--output', '-o', 
                              help='Fichier de sortie (JSON ou YAML)')
    export_parser.set_defaults(func=export_config)
    
    # Commande diff
    diff_parser = subparsers.add_parser('diff', help='Comparer deux environnements')
    diff_parser.add_argument('--env1', required=True, help='Premier environnement')
    diff_parser.add_argument('--env2', required=True, help='Deuxième environnement')
    diff_parser.set_defaults(func=check_differences)
    
    # Commande init
    init_parser = subparsers.add_parser('init', help='Initialiser la structure de configuration')
    init_parser.add_argument('--environment', '-e', default='development',
                            help='Environnement initial')
    init_parser.set_defaults(func=init_config)
    
    # Parser les arguments
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return
    
    # Exécuter la commande
    try:
        args.func(args)
    except KeyboardInterrupt:
        print("\n⚠️ Interruption par l'utilisateur")
    except Exception as e:
        print(f"❌ Erreur inattendue: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()