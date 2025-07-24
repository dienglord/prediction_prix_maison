#!/usr/bin/env python3
"""
Module d'ingénierie des variables pour prédiction prix maisons
Transforme les données nettoyées en variables enrichies pour ML
Auteur: [Votre Nom]
Date: 2024
"""

import hydra
from omegaconf import DictConfig
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class IngenieurVariables:
    """
    Classe pour créer des variables enrichies à partir des données nettoyées
    Applique le Feature Engineering selon les meilleures pratiques ML
    """
    
    def __init__(self, config: DictConfig):
        self.config = config
        self.chemin_processed = Path(config.chemins.data.processed)
        self.annee_reference = 2024  # Pour calculer l'âge des propriétés
        
        # Encoders qui seront réutilisés
        self.label_encoders = {}
        self.scaler = None
        
    def charger_donnees_nettoyees(self):
        """Charge les données depuis l'étape de nettoyage"""
        chemin = self.chemin_processed / "donnees_nettoyees.csv"
        
        if not chemin.exists():
            raise FileNotFoundError(
                f"Fichier non trouvé: {chemin}\n"
                "Exécutez d'abord: python src/data/nettoyeur_donnees.py"
            )
        
        donnees = pd.read_csv(chemin)
        logger.info(f"📊 Données chargées pour feature engineering: {len(donnees)} lignes, {len(donnees.columns)} colonnes")
        
        return donnees
    
    def appliquer_ingenierie_complete(self, donnees: pd.DataFrame):
        """
        Pipeline complet d'ingénierie des variables
        Transforme les données brutes en variables ML-ready
        """
        logger.info("🔧 Démarrage ingénierie des variables")
        
        # Copie pour éviter de modifier l'original
        donnees_enriched = donnees.copy()
        
        # 1. Variables temporelles (âge, rénovation)
        donnees_enriched = self._creer_variables_temporelles(donnees_enriched)
        
        # 2. Variables de superficie et ratios
        donnees_enriched = self._creer_variables_superficie(donnees_enriched)
        
        # 3. Variables de qualité et confort
        donnees_enriched = self._creer_variables_qualite(donnees_enriched)
        
        # 4. Variables de localisation
        donnees_enriched = self._creer_variables_localisation(donnees_enriched)
        
        # 5. Variables d'interaction
        donnees_enriched = self._creer_variables_interaction(donnees_enriched)
        
        # 6. Encodage des variables catégorielles
        donnees_enriched = self._encoder_variables_categorielles(donnees_enriched)
        
        # 7. Nettoyage final
        donnees_enriched = self._nettoyer_variables_finales(donnees_enriched)
        
        logger.info(f"✅ Ingénierie terminée: {len(donnees_enriched.columns)} variables créées")
        
        return donnees_enriched
    
    def _creer_variables_temporelles(self, donnees: pd.DataFrame):
        """Crée des variables liées au temps et à l'âge"""
        logger.info("📅 Création variables temporelles")
        
        # Âge de la propriété
        if 'yr_built' in donnees.columns:
            donnees['age_propriete'] = self.annee_reference - donnees['yr_built']
            logger.info("   ✓ age_propriete créée")
        
        # Années depuis rénovation (si existe)
        if 'yr_renovated' in donnees.columns:
            # Si pas rénové (0), mettre l'âge de la propriété
            donnees['annees_depuis_renovation'] = np.where(
                donnees['yr_renovated'] > 0,
                self.annee_reference - donnees['yr_renovated'],
                donnees.get('age_propriete', 0)
            )
            
            # Variable binaire: a été rénové
            donnees['a_ete_renovee'] = (donnees['yr_renovated'] > 0).astype(int)
            logger.info("   ✓ annees_depuis_renovation et a_ete_renovee créées")
        
        # Catégories d'âge
        if 'age_propriete' in donnees.columns:
            donnees['categorie_age'] = pd.cut(
                donnees['age_propriete'],
                bins=[0, 10, 30, 50, 100, 200],
                labels=['Neuve', 'Récente', 'Moyenne', 'Ancienne', 'Très_ancienne'],
                include_lowest=True
            )
            logger.info("   ✓ categorie_age créée")
        
        return donnees
    
    def _creer_variables_superficie(self, donnees: pd.DataFrame):
        """Crée des variables liées aux superficies et ratios"""
        logger.info("📐 Création variables superficie")
        
        # Prix par pied carré (si price existe)
        if 'price' in donnees.columns and 'sqft_living' in donnees.columns:
            donnees['prix_par_pied_carre'] = donnees['price'] / donnees['sqft_living']
            logger.info("   ✓ prix_par_pied_carre créée")
        
        # Superficie totale (intérieur + sous-sol)
        colonnes_superficie = ['sqft_living', 'sqft_basement']
        if all(col in donnees.columns for col in colonnes_superficie):
            donnees['superficie_totale'] = donnees['sqft_living'] + donnees['sqft_basement']
            logger.info("   ✓ superficie_totale créée")
        
        # Ratio intérieur/terrain
        if 'sqft_living' in donnees.columns and 'sqft_lot' in donnees.columns:
            donnees['ratio_interieur_terrain'] = donnees['sqft_living'] / donnees['sqft_lot']
            logger.info("   ✓ ratio_interieur_terrain créée")
        
        # Superficie par chambre
        if 'sqft_living' in donnees.columns and 'bedrooms' in donnees.columns:
            donnees['superficie_par_chambre'] = np.where(
                donnees['bedrooms'] > 0,
                donnees['sqft_living'] / donnees['bedrooms'],
                donnees['sqft_living']  # Si 0 chambre, prendre toute la superficie
            )
            logger.info("   ✓ superficie_par_chambre créée")
        
        # Catégories de superficie
        if 'sqft_living' in donnees.columns:
            donnees['categorie_superficie'] = pd.cut(
                donnees['sqft_living'],
                bins=[0, 1000, 2000, 3000, 5000, float('inf')],
                labels=['Petit', 'Moyen', 'Grand', 'Très_grand', 'Exceptionnel'],
                include_lowest=True
            )
            logger.info("   ✓ categorie_superficie créée")
        
        return donnees
    
    def _creer_variables_qualite(self, donnees: pd.DataFrame):
        """Crée des variables de qualité et confort"""
        logger.info("⭐ Création variables qualité")
        
        # Ratio salles de bain par chambre
        if 'bathrooms' in donnees.columns and 'bedrooms' in donnees.columns:
            donnees['ratio_sdb_par_chambre'] = np.where(
                donnees['bedrooms'] > 0,
                donnees['bathrooms'] / donnees['bedrooms'],
                donnees['bathrooms']
            )
            logger.info("   ✓ ratio_sdb_par_chambre créée")
        
        # Score de confort (chambres + sdb)
        if 'bedrooms' in donnees.columns and 'bathrooms' in donnees.columns:
            donnees['score_confort'] = donnees['bedrooms'] + donnees['bathrooms']
            logger.info("   ✓ score_confort créée")
        
        # A un sous-sol
        if 'sqft_basement' in donnees.columns:
            donnees['a_sous_sol'] = (donnees['sqft_basement'] > 0).astype(int)
            logger.info("   ✓ a_sous_sol créée")
        
        # Score de qualité combinée
        colonnes_qualite = ['condition', 'grade']
        if all(col in donnees.columns for col in colonnes_qualite):
            # Normaliser sur 10
            condition_norm = (donnees['condition'] - 1) / 4 * 10  # condition 1-5 → 0-10
            grade_norm = (donnees['grade'] - donnees['grade'].min()) / (donnees['grade'].max() - donnees['grade'].min()) * 10
            
            donnees['score_qualite_global'] = (condition_norm + grade_norm) / 2
            logger.info("   ✓ score_qualite_global créée")
        
        # Vue premium (waterfront ou view > 0)
        colonnes_vue = ['waterfront', 'view']
        if all(col in donnees.columns for col in colonnes_vue):
            donnees['vue_premium'] = ((donnees['waterfront'] == 1) | (donnees['view'] > 0)).astype(int)
            logger.info("   ✓ vue_premium créée")
        
        return donnees
    
    def _creer_variables_localisation(self, donnees: pd.DataFrame):
        """Crée des variables liées à la localisation"""
        logger.info("🗺️ Création variables localisation")
        
        # Encoder les villes principales (one-hot encoding)
        if 'city' in donnees.columns:
            # Garder seulement les 10 villes les plus fréquentes
            top_cities = donnees['city'].value_counts().head(10).index.tolist()
            
            for city in top_cities:
                donnees[f'ville_{city}'] = (donnees['city'] == city).astype(int)
            
            logger.info(f"   ✓ {len(top_cities)} variables ville_* créées")
        
        # Zipcode en catégories (par millier)
        if 'zipcode' in donnees.columns:
            donnees['zone_zipcode'] = (donnees['zipcode'] // 1000).astype(str)
            logger.info("   ✓ zone_zipcode créée")
        
        return donnees
    
    def _creer_variables_interaction(self, donnees: pd.DataFrame):
        """Crée des variables d'interaction entre features"""
        logger.info("🔗 Création variables d'interaction")
        
        # Interaction âge x qualité
        if 'age_propriete' in donnees.columns and 'condition' in donnees.columns:
            donnees['age_x_qualite'] = donnees['age_propriete'] * donnees['condition']
            logger.info("   ✓ age_x_qualite créée")
        
        # Interaction superficie x étages
        if 'sqft_living' in donnees.columns and 'floors' in donnees.columns:
            donnees['superficie_x_etages'] = donnees['sqft_living'] * donnees['floors']
            logger.info("   ✓ superficie_x_etages créée")
        
        # Luxe score (vue + qualité + superficie)
        colonnes_luxe = ['vue_premium', 'grade', 'sqft_living']
        if all(col in donnees.columns for col in ['vue_premium', 'grade', 'sqft_living']):
            donnees['score_luxe'] = (
                donnees['vue_premium'] * 1000 +
                donnees['grade'] * 100 +
                donnees['sqft_living'] / 10
            )
            logger.info("   ✓ score_luxe créée")
        
        return donnees
    
    def _encoder_variables_categorielles(self, donnees: pd.DataFrame):
        """Encode les variables catégorielles"""
        logger.info("🏷️ Encodage variables catégorielles")
        
        # Colonnes catégorielles à encoder
        colonnes_cat = donnees.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Exclure les colonnes déjà traitées en one-hot
        colonnes_cat = [col for col in colonnes_cat if not col.startswith('ville_')]
        
        for col in colonnes_cat:
            if col in ['city', 'zone_zipcode']:  # Ces colonnes importantes
                # Label encoding pour garder l'information ordinale si elle existe
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    donnees[f'{col}_encoded'] = self.label_encoders[col].fit_transform(donnees[col].astype(str))
                else:
                    donnees[f'{col}_encoded'] = self.label_encoders[col].transform(donnees[col].astype(str))
                
                logger.info(f"   ✓ {col}_encoded créée")
        
        return donnees
    
    def _nettoyer_variables_finales(self, donnees: pd.DataFrame):
        """Nettoyage final et validation des variables créées"""
        logger.info("🧹 Nettoyage final des variables")
        
        # Remplacer les valeurs infinies par NaN puis par la médiane
        donnees = donnees.replace([np.inf, -np.inf], np.nan)
        
        # Pour les colonnes numériques avec NaN, remplacer par la médiane
        colonnes_numeriques = donnees.select_dtypes(include=[np.number]).columns
        for col in colonnes_numeriques:
            if donnees[col].isnull().any():
                mediane = donnees[col].median()
                donnees[col].fillna(mediane, inplace=True)
                logger.info(f"   ✓ NaN remplacés dans {col}")
        
        # Supprimer les colonnes originales devenues inutiles
        colonnes_a_supprimer = [
            'date',  # Si elle existe
            'id',    # ID pas utile pour ML
            'city',  # Remplacée par ville_* et city_encoded
            'zipcode'  # Remplacée par zone_zipcode
        ]
        
        colonnes_existantes_a_supprimer = [col for col in colonnes_a_supprimer if col in donnees.columns]
        if colonnes_existantes_a_supprimer:
            donnees = donnees.drop(columns=colonnes_existantes_a_supprimer)
            logger.info(f"   ✓ Colonnes supprimées: {colonnes_existantes_a_supprimer}")
        
        # Convertir les variables catégorielles en string pour éviter les erreurs
        colonnes_cat = donnees.select_dtypes(include=['category']).columns
        for col in colonnes_cat:
            donnees[col] = donnees[col].astype(str)
        
        return donnees
    
    def sauvegarder_donnees_enrichies(self, donnees: pd.DataFrame):
        """Sauvegarde les données avec variables enrichies"""
        self.chemin_processed.mkdir(parents=True, exist_ok=True)
        chemin_sortie = self.chemin_processed / "donnees_avec_variables.csv"
        
        donnees.to_csv(chemin_sortie, index=False)
        logger.info(f"💾 Données enrichies sauvegardées: {chemin_sortie}")
        
        # Créer un rapport des variables créées
        self._creer_rapport_variables(donnees, chemin_sortie.parent / "rapport_variables.txt")
    
    def _creer_rapport_variables(self, donnees: pd.DataFrame, chemin_rapport: Path):
        """Crée un rapport détaillé des variables créées"""
        with open(chemin_rapport, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("RAPPORT D'INGÉNIERIE DES VARIABLES\n")
            f.write("=" * 60 + "\n")
            f.write(f"Date de génération: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Nombre total de variables: {len(donnees.columns)}\n")
            f.write(f"Nombre d'observations: {len(donnees)}\n\n")
            
            f.write("VARIABLES PAR CATÉGORIE:\n")
            f.write("-" * 30 + "\n")
            
            # Grouper les variables par type
            variables_temporelles = [col for col in donnees.columns if any(x in col.lower() for x in ['age', 'annee', 'renovation'])]
            variables_superficie = [col for col in donnees.columns if any(x in col.lower() for x in ['superficie', 'sqft', 'ratio', 'pied'])]
            variables_qualite = [col for col in donnees.columns if any(x in col.lower() for x in ['score', 'qualite', 'confort', 'vue', 'sous_sol'])]
            variables_localisation = [col for col in donnees.columns if any(x in col.lower() for x in ['ville', 'zone', 'zipcode'])]
            variables_interaction = [col for col in donnees.columns if '_x_' in col or 'luxe' in col.lower()]
            
            categories = [
                ("Temporelles", variables_temporelles),
                ("Superficie", variables_superficie),
                ("Qualité", variables_qualite),
                ("Localisation", variables_localisation),
                ("Interaction", variables_interaction)
            ]
            
            for nom_cat, variables in categories:
                if variables:
                    f.write(f"\n{nom_cat} ({len(variables)}):\n")
                    for var in variables:
                        f.write(f"  - {var}\n")
            
            f.write("\nSTATISTIQUES DESCRIPTIVES:\n")
            f.write("-" * 30 + "\n")
            
            # Statistiques des variables numériques
            variables_num = donnees.select_dtypes(include=[np.number]).columns.tolist()
            if variables_num:
                stats = donnees[variables_num].describe()
                f.write(stats.to_string())
        
        logger.info(f"📊 Rapport des variables créé: {chemin_rapport}")

@hydra.main(config_path="../../conf", config_name="config", version_base="1.1")
def main(config: DictConfig):
    """Fonction principale d'ingénierie des variables"""
    ingenieur = IngenieurVariables(config)
    
    try:
        # Charger données nettoyées
        donnees = ingenieur.charger_donnees_nettoyees()
        
        # Appliquer ingénierie complète
        donnees_enriched = ingenieur.appliquer_ingenierie_complete(donnees)
        
        # Sauvegarder
        ingenieur.sauvegarder_donnees_enrichies(donnees_enriched)
        
        print(f"✅ Ingénierie des variables terminée!")
        print(f"📊 Variables créées: {len(donnees_enriched.columns)}")
        print(f"📈 Observations: {len(donnees_enriched)}")
        
        # Afficher un échantillon des nouvelles variables
        nouvelles_variables = [col for col in donnees_enriched.columns if col not in donnees.columns]
        if nouvelles_variables:
            print(f"\n🆕 Nouvelles variables créées ({len(nouvelles_variables)}):")
            for var in nouvelles_variables[:10]:  # Afficher les 10 premières
                print(f"   • {var}")
            if len(nouvelles_variables) > 10:
                print(f"   ... et {len(nouvelles_variables) - 10} autres")
        
    except Exception as e:
        logger.error(f"❌ Erreur lors de l'ingénierie des variables: {e}")
        raise

if __name__ == "__main__":
    main()