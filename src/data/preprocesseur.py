import pandas as pd
import numpy as np
import logging
from pathlib import Path
import sys
from typing import Optional, Dict, Any, List, Tuple
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

class PreprocesseurDonnees:
    """
    Préprocesseur optimisé pour l'API - Version finale.
    Compatible avec les 11 features simples uniquement.
    """
    
    def __init__(self, racine_projet: Optional[Path] = None):
        if racine_projet is None:
            self.racine_projet = Path(__file__).resolve().parents[2]
        else:
            self.racine_projet = Path(racine_projet)
        
        self.setup_logging()
        
        # Transformateurs simples
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='median')
        self.is_fitted = False
        
        # Features définitives pour l'API
        self.features_api = [
            'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 
            'floors', 'waterfront', 'view', 'condition',
            'sqft_above', 'sqft_basement', 'yr_built'
        ]
        
        self.logger.info(f"Préprocesseur API initialisé. Racine: {self.racine_projet}")
        self.logger.info(f"Features API: {self.features_api}")
    
    def setup_logging(self):
        logs_dir = self.racine_projet / "reports"
        logs_dir.mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(logs_dir / "preprocessing.log", encoding='utf-8'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger("preprocesseur")
    
    def charger_donnees(self, nom_fichier: str, dossier: str = "processed") -> pd.DataFrame:
        chemin_fichier = self.racine_projet / "data" / dossier / nom_fichier
        
        if not chemin_fichier.exists():
            raise FileNotFoundError(f"Fichier non trouvé: {chemin_fichier}")
        
        if nom_fichier.endswith('.csv'):
            df = pd.read_csv(chemin_fichier)
        else:
            raise ValueError(f"Format non supporté: {nom_fichier}")
        
        self.logger.info(f"✅ Données chargées: {df.shape[0]} lignes, {df.shape[1]} colonnes")
        return df
    
    def selectionner_features_api(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Sélectionne uniquement les features utilisées par l'API.
        """
        self.logger.info("🎯 Sélection des features API...")
        
        # Vérifier quelles features existent
        features_disponibles = []
        for feature in self.features_api:
            if feature in df.columns:
                features_disponibles.append(feature)
            else:
                self.logger.warning(f"⚠️ Feature manquante: {feature}")
        
        # Garder seulement ces features + la variable cible si elle existe
        if 'price' in df.columns:
            features_finales = features_disponibles + ['price']
        else:
            features_finales = features_disponibles
        
        df_api = df[features_finales].copy()
        
        self.logger.info(f"✅ Features API sélectionnées: {features_disponibles}")
        self.logger.info(f"📊 Dataset API: {df_api.shape}")
        
        return df_api
    
    def traiter_donnees_api(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Traitement minimal des données pour l'API.
        """
        self.logger.info("🔧 Traitement des données API...")
        
        df_clean = df.copy()
        
        # Traitement spécial pour certaines colonnes
        if 'sqft_basement' in df_clean.columns:
            df_clean['sqft_basement'] = df_clean['sqft_basement'].fillna(0)
        
        # Imputation simple pour les valeurs manquantes
        colonnes_numeriques = df_clean.select_dtypes(include=[np.number]).columns.tolist()
        if 'price' in colonnes_numeriques:
            colonnes_numeriques.remove('price')  # Ne pas imputer la variable cible
        
        if colonnes_numeriques:
            if not self.is_fitted:
                df_clean[colonnes_numeriques] = self.imputer.fit_transform(df_clean[colonnes_numeriques])
            else:
                df_clean[colonnes_numeriques] = self.imputer.transform(df_clean[colonnes_numeriques])
        
        valeurs_manquantes = df_clean.isnull().sum().sum()
        self.logger.info(f"✅ Valeurs manquantes: {valeurs_manquantes}")
        
        return df_clean
    
    def normaliser_features_api(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalisation des features API.
        """
        self.logger.info("📏 Normalisation des features API...")
        
        df_norm = df.copy()
        
        # Identifier les colonnes numériques (sauf price)
        colonnes_a_normaliser = df_norm.select_dtypes(include=[np.number]).columns.tolist()
        if 'price' in colonnes_a_normaliser:
            colonnes_a_normaliser.remove('price')
        
        if colonnes_a_normaliser:
            if not self.is_fitted:
                df_norm[colonnes_a_normaliser] = self.scaler.fit_transform(df_norm[colonnes_a_normaliser])
            else:
                df_norm[colonnes_a_normaliser] = self.scaler.transform(df_norm[colonnes_a_normaliser])
        
        self.logger.info(f"✅ {len(colonnes_a_normaliser)} colonnes normalisées")
        
        return df_norm
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Pipeline complet de preprocessing pour l'entraînement.
        """
        self.logger.info("🚀 Pipeline de preprocessing API - Entraînement...")
        
        # Étapes du pipeline
        df_processed = self.selectionner_features_api(df)
        df_processed = self.traiter_donnees_api(df_processed)
        df_processed = self.normaliser_features_api(df_processed)
        
        self.is_fitted = True
        self.logger.info("✅ Pipeline API terminé")
        
        return df_processed
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transforme de nouvelles données avec le pipeline ajusté.
        Compatible avec l'API (gère les colonnes manquantes).
        """
        if not self.is_fitted:
            raise ValueError("Le préprocesseur doit être ajusté avec fit_transform()")
        
        self.logger.info("🔄 Transformation API...")
        
        # S'assurer qu'on a toutes les features API nécessaires
        df_input = df.copy()
        
        # Ajouter les features manquantes avec des valeurs par défaut
        for feature in self.features_api:
            if feature not in df_input.columns:
                if feature == 'sqft_basement':
                    df_input[feature] = 0
                elif feature == 'waterfront':
                    df_input[feature] = 0
                elif feature == 'view':
                    df_input[feature] = 0
                elif feature == 'condition':
                    df_input[feature] = 3
                else:
                    # Pour les autres features numériques, utiliser la médiane
                    df_input[feature] = df_input[self.features_api].median().get(feature, 0)
        
        # Sélectionner seulement les features API
        df_processed = df_input[self.features_api].copy()
        df_processed = self.traiter_donnees_api(df_processed)
        df_processed = self.normaliser_features_api(df_processed)
        
        self.logger.info("✅ Transformation API terminée")
        
        return df_processed
    
    def diviser_donnees(self, df: pd.DataFrame, test_size: float = 0.2, 
                       validation_size: float = 0.1, random_state: int = 42) -> Tuple[pd.DataFrame, ...]:
        """
        Division stratifiée avancée pour équilibrer les distributions.
        """
        self.logger.info("✂️ Division stratifiée des données...")
        
        X = df.drop(columns=['price'])
        y = df['price']
        
        # Vérifier les distributions initiales
        self.logger.info(f"Distribution y - Mean: {y.mean():.0f}, Std: {y.std():.0f}")
        
        # Stratification avec plus de bins
        try:
            y_binned = pd.qcut(y, q=10, labels=False, duplicates='drop')
            self.logger.info(f"✅ Stratification avec {y_binned.nunique()} bins")
        except ValueError:
            y_binned = pd.qcut(y, q=5, labels=False, duplicates='drop')
            self.logger.info(f"⚠️ Stratification réduite à {y_binned.nunique()} bins")
        
        # Train+Val / Test avec stratification fine
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, 
            stratify=y_binned
        )
        
        # Refaire les bins pour train+val
        try:
            y_temp_binned = pd.qcut(y_temp, q=min(10, y_temp.nunique()//20), labels=False, duplicates='drop')
        except ValueError:
            y_temp_binned = pd.qcut(y_temp, q=5, labels=False, duplicates='drop')
        
        # Train / Val avec stratification
        val_size_adjusted = validation_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state,
            stratify=y_temp_binned
        )
        
        # Vérifier les distributions finales
        self.logger.info(f"📊 Distributions finales:")
        self.logger.info(f"Train - Mean: {y_train.mean():.0f}, Std: {y_train.std():.0f}")
        self.logger.info(f"Val   - Mean: {y_val.mean():.0f}, Std: {y_val.std():.0f}")
        self.logger.info(f"Test  - Mean: {y_test.mean():.0f}, Std: {y_test.std():.0f}")
        
        # Calculer les ratios pour vérifier l'équilibre
        ratio_train_std = y_train.std() / y_train.mean()
        ratio_val_std = y_val.std() / y_val.mean()
        ratio_test_std = y_test.std() / y_test.mean()
        
        self.logger.info(f"📈 Coefficients de variation:")
        self.logger.info(f"Train CV: {ratio_train_std:.3f}")
        self.logger.info(f"Val   CV: {ratio_val_std:.3f}")
        self.logger.info(f"Test  CV: {ratio_test_std:.3f}")
        
        if abs(ratio_test_std - ratio_train_std) > 0.2:
            self.logger.warning("⚠️ Distributions encore déséquilibrées")
        else:
            self.logger.info("✅ Distributions équilibrées")
        
        self.logger.info(f"✅ Division: Train({len(X_train)}) Val({len(X_val)}) Test({len(X_test)})")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def sauvegarder_preprocesseur(self, nom_fichier: str = "preprocesseur.pkl") -> None:
        """Sauvegarde le préprocesseur API."""
        models_dir = self.racine_projet / "models"
        models_dir.mkdir(exist_ok=True)
        
        preprocesseur_data = {
            'scaler': self.scaler,
            'imputer': self.imputer,
            'features_api': self.features_api,
            'is_fitted': self.is_fitted
        }
        
        chemin_fichier = models_dir / nom_fichier
        joblib.dump(preprocesseur_data, chemin_fichier)
        self.logger.info(f"✅ Préprocesseur API sauvegardé: {chemin_fichier}")
    
    def charger_preprocesseur(self, nom_fichier: str = "preprocesseur.pkl") -> None:
        """Charge un préprocesseur API sauvegardé."""
        models_dir = self.racine_projet / "models"
        chemin_fichier = models_dir / nom_fichier
        
        if not chemin_fichier.exists():
            raise FileNotFoundError(f"Préprocesseur non trouvé: {chemin_fichier}")
        
        try:
            preprocesseur_data = joblib.load(chemin_fichier)
            
            self.scaler = preprocesseur_data['scaler']
            self.imputer = preprocesseur_data['imputer']
            self.features_api = preprocesseur_data['features_api']
            self.is_fitted = preprocesseur_data['is_fitted']
            
            self.logger.info(f"✅ Préprocesseur API chargé: {chemin_fichier}")
            self.logger.info(f"Features API: {self.features_api}")
            
        except Exception as e:
            self.logger.error(f"❌ Erreur lors du chargement: {str(e)}")
            raise


def main():
    """Pipeline de preprocessing API-compatible."""
    try:
        print("🚀 === PREPROCESSING API-COMPATIBLE - PHASE 1 ===")
        
        preprocesseur = PreprocesseurDonnees()
        
        print("\n📊 1. Chargement des données...")
        df = preprocesseur.charger_donnees("data_clean.csv", dossier="processed")
        
        print(f"   Colonnes originales: {list(df.columns)}")
        print(f"   Forme: {df.shape}")
        
        print("\n🔧 2. Preprocessing API-compatible...")
        df_processed = preprocesseur.fit_transform(df)
        
        print(f"   Forme après preprocessing: {df_processed.shape}")
        print(f"   Features API: {preprocesseur.features_api}")
        
        print("\n✂️ 3. Division stratifiée...")
        X_train, X_val, X_test, y_train, y_val, y_test = preprocesseur.diviser_donnees(df_processed)
        
        print("\n💾 4. Sauvegarde...")
        processed_dir = preprocesseur.racine_projet / "data" / "processed"
        
        X_train.to_csv(processed_dir / "X_train.csv", index=False)
        X_val.to_csv(processed_dir / "X_val.csv", index=False)
        X_test.to_csv(processed_dir / "X_test.csv", index=False)
        y_train.to_csv(processed_dir / "y_train.csv", index=False)
        y_val.to_csv(processed_dir / "y_val.csv", index=False)
        y_test.to_csv(processed_dir / "y_test.csv", index=False)
        
        preprocesseur.sauvegarder_preprocesseur()
        
        print("\n✅ === PREPROCESSING API-COMPATIBLE TERMINÉ ===")
        print(f"📊 Résumé:")
        print(f"   - Features API: {len(preprocesseur.features_api)}")
        print(f"   - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        print(f"   - Compatible avec l'API REST")
        print(f"   - Gestion automatique des colonnes manquantes")
        
        # Test de la méthode transform pour l'API
        print("\n🧪 Test de transformation API...")
        test_data = pd.DataFrame([{
            'bedrooms': 3.0,
            'bathrooms': 2.0,
            'sqft_living': 1800,
            'sqft_lot': 7500,
            'floors': 2.0,
            'waterfront': 0,
            'view': 0,
            'condition': 3,
            'sqft_above': 1800,
            'sqft_basement': 0,
            'yr_built': 1995
        }])
        
        test_transformed = preprocesseur.transform(test_data)
        print(f"✅ Test API réussi - Shape: {test_transformed.shape}")
        
        return preprocesseur
        
    except Exception as e:
        print(f"❌ Erreur: {str(e)}")
        return None


if __name__ == "__main__":
    resultats = main()