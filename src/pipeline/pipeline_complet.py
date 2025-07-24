import sys
from pathlib import Path
import logging
from datetime import datetime

# Ajouter le chemin src au Python path
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parents[1]
sys.path.append(str(project_root))

# Imports des modules du projet
from src.data.collecteur_donnees import CollecteurDonnees
from src.data.preprocesseur import PreprocesseurDonnees
from src.models.entraineur import EntraineurModeles

class PipelineComplet:
    """
    Pipeline complet pour le projet de prédiction de prix de maisons.
    Phase 1 - Baseline simple et efficace.
    """
    
    def __init__(self, racine_projet: Path = None):
        """
        Initialise le pipeline complet.
        
        Args:
            racine_projet: Chemin vers la racine du projet
        """
        if racine_projet is None:
            self.racine_projet = Path(__file__).resolve().parents[2]
        else:
            self.racine_projet = Path(racine_projet)
        
        # Setup logging
        self.setup_logging()
        
        # Initialiser les composants
        self.collecteur = None
        self.preprocesseur = None
        self.entraineur = None
        
        self.logger.info(f"🚀 Pipeline initialisé - Racine: {self.racine_projet}")
    
    def setup_logging(self):
        """Configure le système de logging global."""
        logs_dir = self.racine_projet / "reports"
        logs_dir.mkdir(exist_ok=True)
        
        # Nom de fichier avec timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"pipeline_complet_{timestamp}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(logs_dir / log_filename, encoding='utf-8'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        self.logger = logging.getLogger("pipeline_complet")
    
    def etape_collecte_donnees(self, nom_fichier_raw: str = "data.csv") -> bool:
        """
        Étape 1: Collecte et nettoyage des données.
        
        Args:
            nom_fichier_raw: Nom du fichier de données brutes
            
        Returns:
            bool: Succès de l'étape
        """
        self.logger.info("=" * 50)
        self.logger.info("📊 ÉTAPE 1: COLLECTE ET NETTOYAGE DES DONNÉES")
        self.logger.info("=" * 50)
        
        try:
            # Initialiser le collecteur
            self.collecteur = CollecteurDonnees(self.racine_projet)
            
            # Charger les données brutes
            self.logger.info(f"Chargement des données: {nom_fichier_raw}")
            donnees_brutes = self.collecteur.charger_donnees_raw(nom_fichier_raw, "csv")
            
            # Analyser les données
            analyse = self.collecteur.analyser_donnees(donnees_brutes)
            self.logger.info(f"Données brutes: {analyse['nb_lignes']} lignes, {analyse['nb_colonnes']} colonnes")
            
            # Nettoyer les données
            donnees_clean = self.collecteur.nettoyer_donnees(donnees_brutes)
            
            # Sauvegarder les données nettoyées
            self.collecteur.sauvegarder_donnees(donnees_clean, "data_clean.csv")
            
            self.logger.info("✅ Étape 1 terminée avec succès")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Erreur dans l'étape 1: {str(e)}")
            return False
    
    def etape_preprocessing(self) -> bool:
        """
        Étape 2: Préprocessing et feature engineering.
        
        Returns:
            bool: Succès de l'étape
        """
        self.logger.info("=" * 50)
        self.logger.info("🔧 ÉTAPE 2: PREPROCESSING ET FEATURE ENGINEERING")
        self.logger.info("=" * 50)
        
        try:
            # Initialiser le préprocesseur
            self.preprocesseur = PreprocesseurDonnees(self.racine_projet)
            
            # Charger les données nettoyées
            self.logger.info("Chargement des données nettoyées...")
            donnees_clean = self.preprocesseur.charger_donnees("data_clean.csv", "processed")
            
            # Preprocessing complet
            self.logger.info("Preprocessing des données...")
            donnees_preprocessees = self.preprocesseur.fit_transform(donnees_clean, "price")
            
            # Division des données
            self.logger.info("Division des données...")
            X_train, X_val, X_test, y_train, y_val, y_test = self.preprocesseur.diviser_donnees(donnees_preprocessees)
            
            # Sauvegarder tout
            self.logger.info("Sauvegarde des datasets...")
            processed_dir = self.racine_projet / "data" / "processed"
            
            donnees_preprocessees.to_csv(processed_dir / "donnees_preprocessees.csv", index=False)
            X_train.to_csv(processed_dir / "X_train.csv", index=False)
            X_val.to_csv(processed_dir / "X_val.csv", index=False)
            X_test.to_csv(processed_dir / "X_test.csv", index=False)
            y_train.to_csv(processed_dir / "y_train.csv", index=False)
            y_val.to_csv(processed_dir / "y_val.csv", index=False)
            y_test.to_csv(processed_dir / "y_test.csv", index=False)
            
            # Sauvegarder le préprocesseur
            self.preprocesseur.sauvegarder_preprocesseur()
            
            self.logger.info(f"✅ Étape 2 terminée: {donnees_preprocessees.shape} → Train({len(X_train)}) Val({len(X_val)}) Test({len(X_test)})")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Erreur dans l'étape 2: {str(e)}")
            return False
    
    def etape_entrainement(self) -> bool:
        """
        Étape 3: Entraînement et évaluation des modèles.
        
        Returns:
            bool: Succès de l'étape
        """
        self.logger.info("=" * 50)
        self.logger.info("🤖 ÉTAPE 3: ENTRAÎNEMENT ET ÉVALUATION")
        self.logger.info("=" * 50)
        
        try:
            # Initialiser l'entraîneur
            self.entraineur = EntraineurModeles(self.racine_projet)
            
            # Charger les données préprocessées
            self.logger.info("Chargement des données préprocessées...")
            X_train, X_val, X_test, y_train, y_val, y_test = self.entraineur.charger_donnees_preprocessees()
            
            # Entraîner tous les modèles
            self.logger.info("Entraînement de tous les modèles...")
            resultats = self.entraineur.entrainer_tous_modeles(X_train, y_train, X_val, y_val)
            
            # Évaluation finale sur le test
            self.logger.info("Évaluation finale...")
            metriques_test = self.entraineur.evaluer_sur_test(X_test, y_test)
            
            # Créer le rapport de comparaison
            rapport = self.entraineur.creer_rapport_comparaison()
            
            # Sauvegarder le modèle et les résultats
            self.logger.info("Sauvegarde des résultats...")
            self.entraineur.sauvegarder_modele()
            self.entraineur.creer_visualisations()
            
            # Sauvegarder le rapport
            rapport.to_csv(self.racine_projet / "reports" / "comparaison_modeles.csv", index=False)
            
            self.logger.info(f"✅ Étape 3 terminée: Meilleur modèle = {self.entraineur.meilleur_modele} (R² = {self.entraineur.meilleur_score:.4f})")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Erreur dans l'étape 3: {str(e)}")
            return False
    
    def creer_rapport_final(self) -> None:
        """Crée un rapport final du pipeline."""
        self.logger.info("=" * 50)
        self.logger.info("📝 CRÉATION DU RAPPORT FINAL")
        self.logger.info("=" * 50)
        
        try:
            rapport_final = []
            rapport_final.append("# RAPPORT FINAL - PIPELINE PRÉDICTION PRIX MAISONS")
            rapport_final.append("## Phase 1 - Baseline")
            rapport_final.append("")
            rapport_final.append(f"**Date d'exécution:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            rapport_final.append("")
            
            # Informations sur les données
            if self.collecteur:
                rapport_final.append("## 📊 DONNÉES")
                # Ici on pourrait ajouter plus de statistiques
                rapport_final.append("- Source: data.csv")
                rapport_final.append("- Nettoyage: Valeurs manquantes et doublons traités")
                rapport_final.append("")
            
            # Informations sur le preprocessing
            if self.preprocesseur:
                rapport_final.append("## 🔧 PREPROCESSING")
                rapport_final.append(f"- Colonnes numériques: {len(self.preprocesseur.colonnes_numeriques)}")
                rapport_final.append(f"- Colonnes catégorielles (Label): {len(self.preprocesseur.colonnes_categoriques)}")
                rapport_final.append(f"- Colonnes catégorielles (One-Hot): {len(self.preprocesseur.colonnes_a_encoder)}")
                rapport_final.append("- Feature engineering: Prix/m², âge maison, ratios")
                rapport_final.append("- Normalisation: StandardScaler")
                rapport_final.append("")
            
            # Informations sur l'entraînement
            if self.entraineur:
                rapport_final.append("## 🤖 MODÈLES")
                rapport_final.append(f"- Modèles testés: {len(self.entraineur.modeles)}")
                rapport_final.append(f"- Meilleur modèle: {self.entraineur.meilleur_modele}")
                rapport_final.append(f"- Score R² validation: {self.entraineur.meilleur_score:.4f}")
                rapport_final.append("")
            
            # Instructions d'utilisation
            rapport_final.append("## 🚀 UTILISATION")
            rapport_final.append("1. **API:** `python src/api/app.py`")
            rapport_final.append("2. **Documentation:** http://localhost:8000/docs")
            rapport_final.append("3. **Prédiction:** POST /predict avec les features")
            rapport_final.append("")
            
            # Fichiers générés
            rapport_final.append("## 📁 FICHIERS GÉNÉRÉS")
            rapport_final.append("- `models/modele_*.pkl` - Meilleur modèle")
            rapport_final.append("- `models/preprocesseur.pkl` - Préprocesseur")
            rapport_final.append("- `data/processed/` - Datasets préprocessés")
            rapport_final.append("- `reports/` - Logs et visualisations")
            rapport_final.append("")
            
            # Prochaines étapes
            rapport_final.append("## 🔮 PHASE 2 - MLOPS")
            rapport_final.append("- Configuration YAML")
            rapport_final.append("- Tracking MLflow")
            rapport_final.append("- Tests automatisés")
            rapport_final.append("- CI/CD GitHub Actions")
            rapport_final.append("- Monitoring en production")
            rapport_final.append("- Containerisation Docker")
            
            # Sauvegarder le rapport
            rapport_path = self.racine_projet / "reports" / "rapport_final.md"
            with open(rapport_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(rapport_final))
            
            self.logger.info(f"✅ Rapport final créé: {rapport_path}")
            
        except Exception as e:
            self.logger.error(f"❌ Erreur lors de la création du rapport: {str(e)}")
    
    def executer_pipeline_complet(self, nom_fichier_raw: str = "data.csv") -> bool:
        """
        Exécute le pipeline complet de bout en bout.
        
        Args:
            nom_fichier_raw: Nom du fichier de données brutes
            
        Returns:
            bool: Succès du pipeline complet
        """
        self.logger.info("🚀" * 20)
        self.logger.info("🚀 DÉMARRAGE DU PIPELINE COMPLET - PHASE 1 BASELINE")
        self.logger.info("🚀" * 20)
        
        debut = datetime.now()
        
        # Étape 1: Collecte des données
        if not self.etape_collecte_donnees(nom_fichier_raw):
            self.logger.error("❌ Échec de l'étape 1")
            return False
        
        # Étape 2: Preprocessing
        if not self.etape_preprocessing():
            self.logger.error("❌ Échec de l'étape 2")
            return False
        
        # Étape 3: Entraînement
        if not self.etape_entrainement():
            self.logger.error("❌ Échec de l'étape 3")
            return False
        
        # Créer le rapport final
        self.creer_rapport_final()
        
        # Résumé final
        fin = datetime.now()
        duree = fin - debut
        
        self.logger.info("🎉" * 20)
        self.logger.info("🎉 PIPELINE TERMINÉ AVEC SUCCÈS!")
        self.logger.info("🎉" * 20)
        self.logger.info(f"⏱️  Durée totale: {duree}")
        self.logger.info(f"🏆 Meilleur modèle: {self.entraineur.meilleur_modele}")
        self.logger.info(f"📊 Score R²: {self.entraineur.meilleur_score:.4f}")
        self.logger.info("")
        self.logger.info("🚀 PROCHAINES ÉTAPES:")
        self.logger.info("   1. Tester l'API: python src/api/app.py")
        self.logger.info("   2. Voir la doc: http://localhost:8000/docs")
        self.logger.info("   3. Phase 2 MLOps!")
        
        return True


def main():
    """Fonction principale pour exécuter le pipeline complet."""
    try:
        print("🚀 === PIPELINE COMPLET - PHASE 1 BASELINE ===")
        print()
        
        # Créer le pipeline
        pipeline = PipelineComplet()
        
        # Exécuter le pipeline complet
        succes = pipeline.executer_pipeline_complet("data.csv")
        
        if succes:
            print("\n✅ === PIPELINE TERMINÉ AVEC SUCCÈS ===")
            print("🎯 Votre modèle est prêt à être utilisé!")
            print("📝 Consultez reports/rapport_final.md pour plus de détails")
            print()
            print("🚀 Pour tester l'API:")
            print("   cd src/api && python app.py")
            print("   Puis allez sur: http://localhost:8000/docs")
            
            return True
        else:
            print("\n❌ === ÉCHEC DU PIPELINE ===")
            print("Consultez les logs dans reports/ pour plus de détails")
            return False
            
    except KeyboardInterrupt:
        print("\n⚠️ Pipeline interrompu par l'utilisateur")
        return False
        
    except Exception as e:
        print(f"\n❌ Erreur fatale: {str(e)}")
        return False


if __name__ == "__main__":
    succes = main()
    sys.exit(0 if succes else 1)