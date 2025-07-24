import pandas as pd
import numpy as np
import logging
from pathlib import Path
import sys
from typing import Optional, Dict, Any
import os

class CollecteurDonnees:
    """
    Classe pour collecter et traiter les données immobilières.
    """
    
    def __init__(self, racine_projet: Optional[Path] = None):
        """
        Initialise le collecteur de données.
        
        Args:
            racine_projet (Path, optional): Chemin vers la racine du projet
        """
        # Configuration du projet
        if racine_projet is None:
            self.racine_projet = Path(__file__).resolve().parents[2]
        else:
            self.racine_projet = Path(racine_projet)
        
        # Configuration du logging
        self.setup_logging()
        
        # Créer les dossiers nécessaires
        self.creer_structure_dossiers()
        
        self.logger.info(f"Collecteur initialisé. Racine du projet: {self.racine_projet}")
    
    def setup_logging(self):
        """Configure le système de logging."""
        # Créer le dossier logs s'il n'existe pas
        logs_dir = self.racine_projet / "logs"
        logs_dir.mkdir(exist_ok=True)
        
        # Configuration du logger
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(logs_dir / "collecteur_donnees.log", encoding='utf-8'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger("**main**")
    
    def creer_structure_dossiers(self):
        """Crée la structure de dossiers nécessaire."""
        dossiers = [
            "data/raw",
            "data/processed",
            "data/interim",
            "logs",
            "models",
            "reports"
        ]
        
        for dossier in dossiers:
            chemin = self.racine_projet / dossier
            chemin.mkdir(parents=True, exist_ok=True)
    
    def charger_donnees_raw(self, nom_fichier: str, format_fichier: str = 'csv') -> pd.DataFrame:
        """
        Charge les données depuis un fichier CSV ou Excel.
        
        Args:
            nom_fichier (str): Nom du fichier à charger
            format_fichier (str): Format du fichier ('csv' ou 'excel')
            
        Returns:
            pd.DataFrame: Les données chargées
            
        Raises:
            FileNotFoundError: Si le fichier n'existe pas
            ValueError: Si le format de fichier n'est pas supporté
        """
        chemin_fichier = self.racine_projet / "data" / "raw" / nom_fichier
        
        if not chemin_fichier.exists():
            raise FileNotFoundError(f"Fichier non trouvé: {chemin_fichier}")
        
        self.logger.info(f"Chargement du fichier: {chemin_fichier}")
        
        try:
            if format_fichier.lower() == 'csv':
                # Essayer différents encodages pour les fichiers CSV
                encodages = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
                df = None
                
                for encoding in encodages:
                    try:
                        df = pd.read_csv(chemin_fichier, encoding=encoding)
                        self.logger.info(f"✅ Fichier CSV chargé avec l'encodage: {encoding}")
                        break
                    except UnicodeDecodeError:
                        continue
                
                if df is None:
                    raise ValueError("Impossible de décoder le fichier CSV avec les encodages testés")
                    
            elif format_fichier.lower() == 'excel':
                df = pd.read_excel(chemin_fichier)
                self.logger.info("✅ Fichier Excel chargé")
                
            else:
                raise ValueError(f"Format de fichier non supporté: {format_fichier}. Formats acceptés: 'csv', 'excel'")
            
            self.logger.info(f"✅ Données chargées: {df.shape[0]} lignes, {df.shape[1]} colonnes")
            
            # Afficher les premières lignes et informations de base
            self.logger.info(f"Colonnes disponibles: {list(df.columns)}")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Erreur lors du chargement: {str(e)}")
            raise
    
    def nettoyer_donnees(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Nettoie les données chargées.
        
        Args:
            df (pd.DataFrame): DataFrame à nettoyer
            
        Returns:
            pd.DataFrame: DataFrame nettoyé
        """
        self.logger.info("🧹 Début du nettoyage des données...")
        
        df_clean = df.copy()
        
        # Statistiques avant nettoyage
        nb_lignes_initial = len(df_clean)
        nb_doublons = df_clean.duplicated().sum()
        
        self.logger.info(f"Données avant nettoyage: {nb_lignes_initial} lignes")
        self.logger.info(f"Doublons détectés: {nb_doublons}")
        
        # Supprimer les doublons
        df_clean = df_clean.drop_duplicates()
        
        # Supprimer les lignes avec trop de valeurs manquantes (plus de 50%)
        seuil_na = len(df_clean.columns) * 0.5
        df_clean = df_clean.dropna(thresh=seuil_na)
        
        # Nettoyer les noms de colonnes
        df_clean.columns = df_clean.columns.str.strip().str.lower().str.replace(' ', '_')
        
        # Statistiques après nettoyage
        nb_lignes_final = len(df_clean)
        lignes_supprimees = nb_lignes_initial - nb_lignes_final
        
        self.logger.info(f"✅ Nettoyage terminé:")
        self.logger.info(f"  - Lignes supprimées: {lignes_supprimees}")
        self.logger.info(f"  - Lignes restantes: {nb_lignes_final}")
        
        return df_clean
    
    def sauvegarder_donnees(self, df: pd.DataFrame, nom_fichier: str, dossier: str = "processed") -> None:
        """
        Sauvegarde les données dans un fichier.
        
        Args:
            df (pd.DataFrame): DataFrame à sauvegarder
            nom_fichier (str): Nom du fichier de sortie
            dossier (str): Dossier de destination (processed, interim, etc.)
        """
        chemin_sortie = self.racine_projet / "data" / dossier / nom_fichier
        
        try:
            # Créer le dossier s'il n'existe pas
            chemin_sortie.parent.mkdir(parents=True, exist_ok=True)
            
            # Sauvegarder selon l'extension
            if nom_fichier.endswith('.csv'):
                df.to_csv(chemin_sortie, index=False, encoding='utf-8')
            elif nom_fichier.endswith('.xlsx'):
                df.to_excel(chemin_sortie, index=False)
            else:
                raise ValueError(f"Format de fichier non supporté pour la sauvegarde: {nom_fichier}")
            
            self.logger.info(f"✅ Données sauvegardées: {chemin_sortie}")
            
        except Exception as e:
            self.logger.error(f"❌ Erreur lors de la sauvegarde: {str(e)}")
            raise
    
    def analyser_donnees(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyse les données et retourne des statistiques descriptives.
        
        Args:
            df (pd.DataFrame): DataFrame à analyser
            
        Returns:
            Dict[str, Any]: Statistiques descriptives
        """
        self.logger.info("📊 Analyse des données...")
        
        analyse = {
            'nb_lignes': len(df),
            'nb_colonnes': len(df.columns),
            'colonnes': list(df.columns),
            'types_colonnes': df.dtypes.to_dict(),
            'valeurs_manquantes': df.isnull().sum().to_dict(),
            'statistiques_numeriques': df.describe().to_dict() if len(df.select_dtypes(include=[np.number]).columns) > 0 else {}
        }
        
        self.logger.info(f"✅ Analyse terminée:")
        self.logger.info(f"  - {analyse['nb_lignes']} lignes")
        self.logger.info(f"  - {analyse['nb_colonnes']} colonnes")
        self.logger.info(f"  - Colonnes: {analyse['colonnes']}")
        
        return analyse
    
    def traitement_complet(self, nom_fichier_entree: str, nom_fichier_sortie: str = None) -> pd.DataFrame:
        """
        Effectue le traitement complet des données.
        
        Args:
            nom_fichier_entree (str): Nom du fichier d'entrée
            nom_fichier_sortie (str, optional): Nom du fichier de sortie
            
        Returns:
            pd.DataFrame: Données traitées
        """
        self.logger.info("🚀 Début du traitement complet...")
        
        # 1. Charger les données
        df_raw = self.charger_donnees_raw(nom_fichier_entree)
        
        # 2. Analyser les données brutes
        analyse_raw = self.analyser_donnees(df_raw)
        
        # 3. Nettoyer les données
        df_clean = self.nettoyer_donnees(df_raw)
        
        # 4. Analyser les données nettoyées
        analyse_clean = self.analyser_donnees(df_clean)
        
        # 5. Sauvegarder les données nettoyées
        if nom_fichier_sortie:
            self.sauvegarder_donnees(df_clean, nom_fichier_sortie)
        
        self.logger.info("✅ Traitement complet terminé!")
        
        return df_clean


def main():
    """Fonction principale pour tester le collecteur."""
    try:
        # Initialiser le collecteur
        collecteur = CollecteurDonnees()
        
        print("🚀 Chargement des données...")
        
        # Charger et traiter les données
        # Spécifiez explicitement le format du fichier
        donnees = collecteur.charger_donnees_raw("data.csv", "csv")  # Pour CSV
        # donnees = collecteur.charger_donnees_raw("data.xlsx", "excel")  # Pour Excel
        
        print("📊 Analyse des données...")
        analyse = collecteur.analyser_donnees(donnees)
        
        print("🧹 Nettoyage des données...")
        donnees_clean = collecteur.nettoyer_donnees(donnees)
        
        print("💾 Sauvegarde des données nettoyées...")
        collecteur.sauvegarder_donnees(donnees_clean, "data_clean.csv")
        
        print("✅ Traitement terminé avec succès!")
        print(f"📈 Résumé: {len(donnees_clean)} lignes, {len(donnees_clean.columns)} colonnes")
        
        return donnees_clean
        
    except FileNotFoundError as e:
        collecteur.logger.error(f"❌ Erreur: {str(e)}")
        print(f"❌ Erreur: Fichier non trouvé. Vérifiez que le fichier existe dans le dossier data/raw/")
        return None
        
    except Exception as e:
        collecteur.logger.error(f"❌ Erreur: {str(e)}")
        print(f"❌ Erreur inattendue: {str(e)}")
        return None


if __name__ == "__main__":
    donnees_traitees = main()