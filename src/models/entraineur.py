import pandas as pd
import numpy as np
import logging
from pathlib import Path
import sys
import joblib
from typing import Dict, Any, Tuple, List
import warnings
warnings.filterwarnings('ignore')

# Modèles ML
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns

class EntraineurModeles:
    """
    Classe pour l'entraînement et l'évaluation de modèles ML.
    Compatible avec la structure Cookiecutter Data Science.
    VERSION FINALE : 3 modèles régularisés pour éviter le surapprentissage.
    """
    
    def __init__(self, racine_projet: Path = None):
        """
        Initialise l'entraîneur de modèles.
        
        Args:
            racine_projet (Path): Chemin vers la racine du projet
        """
        if racine_projet is None:
            self.racine_projet = Path(__file__).resolve().parents[2]
        else:
            self.racine_projet = Path(racine_projet)
        
        # Configuration du logging
        self.setup_logging()
        
        # Dictionnaire des 3 modèles régularisés
        self.modeles = {
            'linear_regression': LinearRegression(),
            
            'random_forest': RandomForestRegressor(
                n_estimators=100,
                max_depth=8,         # Limité pour éviter surapprentissage
                min_samples_split=5, # Au moins 5 échantillons pour diviser
                min_samples_leaf=2,  # Au moins 2 échantillons par feuille
                max_features='sqrt', # Régularisation : sqrt des features
                random_state=42,
                n_jobs=-1
            ),
            
            'gradient_boosting': GradientBoostingRegressor(
                n_estimators=50,     # Réduit pour éviter surapprentissage
                max_depth=4,         # Arbres peu profonds
                learning_rate=0.05,  # Apprentissage lent et stable
                subsample=0.8,       # Échantillonnage aléatoire
                max_features='sqrt', # Régularisation features
                random_state=42
            )
        }
        
        # Stockage des résultats
        self.resultats_entrainement = {}
        self.meilleur_modele = None
        self.meilleur_score = float('-inf')
        self.modele_final = None
        
        self.logger.info(f"Entraîneur initialisé. Racine: {self.racine_projet}")
        self.logger.info(f"✅ 3 modèles régularisés configurés pour éviter le surapprentissage")
        self.logger.info(f"Modèles disponibles: {list(self.modeles.keys())}")
    
    def setup_logging(self):
        """Configure le système de logging."""
        logs_dir = self.racine_projet / "reports"
        logs_dir.mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(logs_dir / "training.log", encoding='utf-8'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger("entraineur")
    
    def charger_donnees_preprocessees(self) -> Tuple[pd.DataFrame, ...]:
        """
        Charge les données préprocessées SIMPLES pour diagnostic.
        
        Returns:
            Tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
        """
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
            
            # Afficher les distributions pour diagnostic
            self.logger.info(f"📊 Distributions des prix:")
            self.logger.info(f"   - Train: Mean={y_train.mean():.0f}, Std={y_train.std():.0f}")
            self.logger.info(f"   - Val:   Mean={y_val.mean():.0f}, Std={y_val.std():.0f}")
            self.logger.info(f"   - Test:  Mean={y_test.mean():.0f}, Std={y_test.std():.0f}")
            
            return X_train, X_val, X_test, y_train, y_val, y_test
            
        except Exception as e:
            self.logger.error(f"❌ Erreur lors du chargement: {str(e)}")
            raise
    
    def calculer_metriques(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calcule les métriques de performance.
        
        Args:
            y_true: Vraies valeurs
            y_pred: Prédictions
            
        Returns:
            Dict: Métriques calculées
        """
        metriques = {
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred),
            'mape': np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100  # Éviter division par 0
        }
        
        return metriques
    
    def entrainer_modele(self, nom_modele: str, X_train: pd.DataFrame, y_train: pd.Series,
                        X_val: pd.DataFrame, y_val: pd.Series) -> Dict[str, Any]:
        """
        Entraîne un modèle spécifique et calcule ses performances.
        
        Args:
            nom_modele: Nom du modèle à entraîner
            X_train: Features d'entraînement
            y_train: Cibles d'entraînement
            X_val: Features de validation
            y_val: Cibles de validation
            
        Returns:
            Dict: Résultats d'entraînement du modèle
        """
        self.logger.info(f"🔧 Entraînement du modèle: {nom_modele}")
        
        if nom_modele not in self.modeles:
            raise ValueError(f"Modèle '{nom_modele}' non disponible")
        
        # Obtenir le modèle
        modele = self.modeles[nom_modele]
        
        try:
            # Entraînement
            modele.fit(X_train, y_train)
            
            # Prédictions
            y_pred_train = modele.predict(X_train)
            y_pred_val = modele.predict(X_val)
            
            # Métriques sur l'ensemble d'entraînement
            metriques_train = self.calculer_metriques(y_train, y_pred_train)
            
            # Métriques sur l'ensemble de validation
            metriques_val = self.calculer_metriques(y_val, y_pred_val)
            
            # Validation croisée sur l'ensemble d'entraînement
            cv_scores = cross_val_score(modele, X_train, y_train, cv=5, 
                                      scoring='neg_mean_squared_error')
            cv_rmse = np.sqrt(-cv_scores)
            
            # Calculer l'écart train/validation (indicateur de surapprentissage)
            ecart_r2 = metriques_train['r2'] - metriques_val['r2']
            
            # Résultats
            resultats = {
                'modele': modele,
                'nom': nom_modele,
                'metriques_train': metriques_train,
                'metriques_val': metriques_val,
                'cv_rmse_mean': cv_rmse.mean(),
                'cv_rmse_std': cv_rmse.std(),
                'ecart_r2': ecart_r2,
                'y_pred_train': y_pred_train,
                'y_pred_val': y_pred_val
            }
            
            # Logging des résultats avec indicateur de surapprentissage
            self.logger.info(f"✅ {nom_modele} entraîné:")
            self.logger.info(f"   - R² validation: {metriques_val['r2']:.4f}")
            self.logger.info(f"   - RMSE validation: {metriques_val['rmse']:.2f}")
            self.logger.info(f"   - CV RMSE: {cv_rmse.mean():.2f} (±{cv_rmse.std():.2f})")
            self.logger.info(f"   - Écart R² train/val: {ecart_r2:.4f} {'⚠️' if ecart_r2 > 0.1 else '✅'}")
            
            return resultats
            
        except Exception as e:
            self.logger.error(f"❌ Erreur lors de l'entraînement de {nom_modele}: {str(e)}")
            raise
    
    def entrainer_tous_modeles(self, X_train: pd.DataFrame, y_train: pd.Series,
                              X_val: pd.DataFrame, y_val: pd.Series) -> Dict[str, Any]:
        """
        Entraîne tous les modèles et compare leurs performances.
        
        Args:
            X_train: Features d'entraînement
            y_train: Cibles d'entraînement
            X_val: Features de validation
            y_val: Cibles de validation
            
        Returns:
            Dict: Résultats de tous les modèles
        """
        self.logger.info("🚀 Entraînement de tous les modèles...")
        
        self.resultats_entrainement = {}
        
        for nom_modele in self.modeles.keys():
            try:
                resultats = self.entrainer_modele(nom_modele, X_train, y_train, X_val, y_val)
                self.resultats_entrainement[nom_modele] = resultats
                
                # Suivre le meilleur modèle (basé sur R² validation)
                r2_val = resultats['metriques_val']['r2']
                if r2_val > self.meilleur_score:
                    self.meilleur_score = r2_val
                    self.meilleur_modele = nom_modele
                    self.modele_final = resultats['modele']
                
            except Exception as e:
                self.logger.error(f"❌ Échec de l'entraînement de {nom_modele}: {str(e)}")
                continue
        
        self.logger.info(f"🏆 Meilleur modèle: {self.meilleur_modele} (R² = {self.meilleur_score:.4f})")
        
        return self.resultats_entrainement
    
    def evaluer_sur_test(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """
        Évalue le meilleur modèle sur l'ensemble de test.
        
        Args:
            X_test: Features de test
            y_test: Cibles de test
            
        Returns:
            Dict: Métriques sur l'ensemble de test
        """
        if self.modele_final is None:
            raise ValueError("Aucun modèle n'a été entraîné")
        
        self.logger.info(f"🎯 Évaluation finale du modèle {self.meilleur_modele} sur le test...")
        
        # Prédictions sur le test
        y_pred_test = self.modele_final.predict(X_test)
        
        # Métriques finales
        metriques_test = self.calculer_metriques(y_test, y_pred_test)
        
        # Calculer l'écart validation/test (indicateur de généralisation)
        r2_val = self.resultats_entrainement[self.meilleur_modele]['metriques_val']['r2']
        ecart_val_test = r2_val - metriques_test['r2']
        
        self.logger.info(f"📊 Performances finales:")
        self.logger.info(f"   - R²: {metriques_test['r2']:.4f}")
        self.logger.info(f"   - RMSE: {metriques_test['rmse']:.2f}")
        self.logger.info(f"   - MAE: {metriques_test['mae']:.2f}")
        self.logger.info(f"   - MAPE: {metriques_test['mape']:.2f}%")
        self.logger.info(f"   - Écart R² val/test: {ecart_val_test:.4f} {'⚠️' if abs(ecart_val_test) > 0.05 else '✅'}")
        
        return metriques_test
    
    def creer_rapport_comparaison(self) -> pd.DataFrame:
        """
        Crée un tableau de comparaison des modèles.
        
        Returns:
            pd.DataFrame: Tableau de comparaison
        """
        if not self.resultats_entrainement:
            raise ValueError("Aucun modèle n'a été entraîné")
        
        # Préparer les données pour le tableau
        donnees_rapport = []
        
        for nom, resultats in self.resultats_entrainement.items():
            donnees_rapport.append({
                'Modèle': nom,
                'R² Train': resultats['metriques_train']['r2'],
                'R² Validation': resultats['metriques_val']['r2'],
                'RMSE Train': resultats['metriques_train']['rmse'],
                'RMSE Validation': resultats['metriques_val']['rmse'],
                'MAE Validation': resultats['metriques_val']['mae'],
                'CV RMSE': resultats['cv_rmse_mean'],
                'CV RMSE Std': resultats['cv_rmse_std'],
                'Écart R² Train/Val': resultats['ecart_r2']
            })
        
        rapport = pd.DataFrame(donnees_rapport)
        rapport = rapport.sort_values('R² Validation', ascending=False)
        
        return rapport
    
    def sauvegarder_modele(self, nom_fichier: str = None) -> None:
        """
        Sauvegarde le meilleur modèle.
        
        Args:
            nom_fichier: Nom du fichier (optionnel)
        """
        if self.modele_final is None:
            raise ValueError("Aucun modèle à sauvegarder")
        
        models_dir = self.racine_projet / "models"
        models_dir.mkdir(exist_ok=True)
        
        if nom_fichier is None:
            nom_fichier = f"modele_{self.meilleur_modele}.pkl"
        
        chemin_fichier = models_dir / nom_fichier
        
        # Données à sauvegarder
        donnees_modele = {
            'modele': self.modele_final,
            'nom_modele': self.meilleur_modele,
            'score': self.meilleur_score,
            'resultats': self.resultats_entrainement[self.meilleur_modele]
        }
        
        joblib.dump(donnees_modele, chemin_fichier)
        self.logger.info(f"✅ Modèle sauvegardé: {chemin_fichier}")
    
    def creer_visualisations(self) -> None:
        """
        Crée des visualisations des résultats.
        """
        if not self.resultats_entrainement:
            raise ValueError("Aucun résultat à visualiser")
        
        figures_dir = self.racine_projet / "reports" / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)
        
        # Style des graphiques
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Comparaison des R² par modèle
        plt.figure(figsize=(12, 6))
        
        modeles_noms = []
        r2_train = []
        r2_val = []
        
        for nom, resultats in self.resultats_entrainement.items():
            modeles_noms.append(nom.replace('_', ' ').title())
            r2_train.append(resultats['metriques_train']['r2'])
            r2_val.append(resultats['metriques_val']['r2'])
        
        x = np.arange(len(modeles_noms))
        width = 0.35
        
        plt.bar(x - width/2, r2_train, width, label='Train', alpha=0.8, color='skyblue')
        plt.bar(x + width/2, r2_val, width, label='Validation', alpha=0.8, color='lightcoral')
        
        plt.xlabel('Modèles')
        plt.ylabel('R² Score')
        plt.title('Comparaison des performances R² par modèle\n(Écart faible = moins de surapprentissage)')
        plt.xticks(x, modeles_noms, rotation=15)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plt.savefig(figures_dir / "comparaison_modeles_r2.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. RMSE par modèle
        plt.figure(figsize=(10, 6))
        
        rmse_train = [self.resultats_entrainement[nom]['metriques_train']['rmse'] for nom in self.resultats_entrainement.keys()]
        rmse_val = [self.resultats_entrainement[nom]['metriques_val']['rmse'] for nom in self.resultats_entrainement.keys()]
        
        x = np.arange(len(modeles_noms))
        
        plt.bar(x - width/2, rmse_train, width, label='Train', alpha=0.8, color='lightgreen')
        plt.bar(x + width/2, rmse_val, width, label='Validation', alpha=0.8, color='orange')
        
        plt.xlabel('Modèles')
        plt.ylabel('RMSE')
        plt.title('Comparaison RMSE par modèle')
        plt.xticks(x, modeles_noms, rotation=15)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plt.savefig(figures_dir / "comparaison_modeles_rmse.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"📊 Visualisations créées dans: {figures_dir}")


def main():
    """Fonction principale - Pipeline d'entraînement baseline."""
    try:
        print("🚀 === ENTRAÎNEMENT MODÈLES - PHASE 1 BASELINE ===")
        
        # Initialiser l'entraîneur
        entraineur = EntraineurModeles()
        
        print("\n📊 1. Chargement des données préprocessées...")
        # Charger les données
        X_train, X_val, X_test, y_train, y_val, y_test = entraineur.charger_donnees_preprocessees()
        
        print(f"   Features: {X_train.shape[1]}")
        print(f"   Échantillons: Train({len(X_train)}) Val({len(X_val)}) Test({len(X_test)})")
        
        print("\n🔧 2. Entraînement des 3 modèles régularisés...")
        print(f"   Modèles: {list(entraineur.modeles.keys())}")
        # Entraîner tous les modèles
        resultats = entraineur.entrainer_tous_modeles(X_train, y_train, X_val, y_val)
        
        print(f"\n🏆 3. Meilleur modèle: {entraineur.meilleur_modele}")
        print(f"   Score R² validation: {entraineur.meilleur_score:.4f}")
        
        print("\n📊 4. Rapport de comparaison:")
        # Créer et afficher le rapport de comparaison
        rapport = entraineur.creer_rapport_comparaison()
        print(rapport.round(4))
        
        print("\n🎯 5. Évaluation finale sur le test...")
        # Évaluer sur l'ensemble de test
        metriques_test = entraineur.evaluer_sur_test(X_test, y_test)
        
        print("\n📊 6. Création des visualisations...")
        # Créer les visualisations
        entraineur.creer_visualisations()
        
        print("\n💾 7. Sauvegarde du meilleur modèle...")
        # Sauvegarder le modèle
        entraineur.sauvegarder_modele()
        
        # Sauvegarder le rapport de comparaison
        reports_dir = entraineur.racine_projet / "reports"
        rapport.to_csv(reports_dir / "comparaison_modeles.csv", index=False)
        
        print("\n✅ === ENTRAÎNEMENT TERMINÉ - PHASE 1 BASELINE ===")
        print(f"🏆 Résultats finaux:")
        print(f"   - Meilleur modèle: {entraineur.meilleur_modele}")
        print(f"   - R² validation: {entraineur.meilleur_score:.4f}")
        print(f"   - R² test: {metriques_test['r2']:.4f}")
        print(f"   - RMSE test: {metriques_test['rmse']:.2f}")
        print(f"   - MAE test: {metriques_test['mae']:.2f}")
        
        # Évaluation de la qualité du modèle
        ecart_val_test = abs(entraineur.meilleur_score - metriques_test['r2'])
        if metriques_test['r2'] > 0.5 and ecart_val_test < 0.05:
            print(f"   ✅ Modèle de bonne qualité (R² > 0.5, écart val/test < 0.05)")
        elif metriques_test['r2'] > 0.3:
            print(f"   🟡 Modèle acceptable (R² > 0.3)")
        else:
            print(f"   ⚠️ Modèle à améliorer (R² < 0.3)")
        
        # Diagnostic supplémentaire
        print(f"\n🔍 DIAGNOSTIC:")
        if ecart_val_test > 0.1:
            print(f"   ⚠️ Écart val/test important ({ecart_val_test:.4f}) - Problème de généralisation")
        else:
            print(f"   ✅ Écart val/test acceptable ({ecart_val_test:.4f})")
            
        print(f"\n📁 Fichiers créés:")
        print(f"   - models/modele_{entraineur.meilleur_modele}.pkl")
        print(f"   - reports/comparaison_modeles.csv")
        print(f"   - reports/figures/comparaisons.png")
        
        return {
            'entraineur': entraineur,
            'resultats': resultats,
            'metriques_test': metriques_test,
            'rapport': rapport
        }
        
    except FileNotFoundError as e:
        print(f"❌ Erreur: {str(e)}")
        print("💡 Assurez-vous d'avoir exécuté le preprocessing d'abord.")
        return None
        
    except Exception as e:
        print(f"❌ Erreur inattendue: {str(e)}")
        return None


if __name__ == "__main__":
    resultats = main()