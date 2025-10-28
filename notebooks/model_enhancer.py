import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Tuple, Optional
import logging
import warnings

# --- Import RDKit and Scikit-learn ---
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit import DataStructs
except ImportError:
    print("FATAL ERROR: RDKit library not found or not properly installed.")
    exit(1)

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib

# --- Configuration ---
DATA_PATH = Path("data/processed/final_combined_dataset.csv")
REPORT_DIR = Path("reports/models")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore', category=UserWarning)

# =======================================================================
# I. Featurization
# =======================================================================

def smiles_to_ecfp4_fingerprint(smiles: str, radius: int = 2, nBits: int = 2048) -> np.ndarray:
    """Convert a SMILES string into an ECFP4 (Morgan fingerprint) binary vector."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.zeros(nBits, dtype=int)
        
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=nBits)
        return np.array(fp, dtype=int)
    except Exception:
        return np.zeros(nBits, dtype=int)

def generate_fingerprint_features(df: pd.DataFrame, smiles_col: str = 'standardized_smiles', nBits: int = 2048) -> np.ndarray:
    """Generate a molecular fingerprint feature matrix for the entire DataFrame."""
    logger.info(f"Starting generation of {nBits}-bit ECFP4 fingerprints...")
    fingerprints = df[smiles_col].apply(
        lambda smi: smiles_to_ecfp4_fingerprint(smi, nBits=nBits)
    ).tolist()
    feature_matrix = np.array(fingerprints)
    return feature_matrix

def generate_combined_features(df: pd.DataFrame) -> Tuple[np.ndarray, list]:
    """Generate combined features: ECFP4 fingerprints plus numeric descriptors."""
    # 1. Fingerprint features (ECFP4)
    X_fp = generate_fingerprint_features(df)
    
    # 2. Numeric descriptors
    descriptor_cols = ['mol_weight', 'logp', 'hbd', 'hba', 'tpsa', 'rotatable_bonds', 'heavy_atoms']
    # Fill missing values and select columns
    X_desc_df = df[descriptor_cols].fillna(0)
    
    # 3. Standardize descriptors (StandardScaler)
    scaler = StandardScaler()
    X_desc_scaled = scaler.fit_transform(X_desc_df.values)
    
    # 4. Concatenate features
    X_combined = np.hstack([X_fp, X_desc_scaled])
    
    logger.info(f"Combined feature matrix shape: {X_combined.shape}")
    
    # 5. Create combined feature names list (for feature importance analysis)
    feature_names = [f"FP_Bit_{i}" for i in range(X_fp.shape[1])] + descriptor_cols
    
    return X_combined, feature_names


# =======================================================================
# II. Model Training and Evaluation
# =======================================================================

def analyze_feature_importance(model: RandomForestRegressor, feature_names: list, n_features: int = 100):
    """
    Extract feature importances from a trained model and select the top-N contributing features.
    """
    if not hasattr(model, 'feature_importances_'):
        logger.warning("Model does not have 'feature_importances_' attribute; skipping feature importance analysis.")
        return

    logger.info(f"\n--- Feature importance analysis (Top {n_features} features) ---")
    
    # 1. Get importances for all features
    importances = pd.Series(model.feature_importances_, index=feature_names)
    
    # 2. Select top N features
    top_N_importances = importances.nlargest(n_features)
    
    logger.info(f"Total number of features: {len(feature_names)}")
    logger.info(f"Top {n_features} most important features and their importances:\n" + top_N_importances.to_string())
    
    # Visualize top N importances
    plt.figure(figsize=(12, 8))
    top_N_importances.plot(kind='barh', color='#ff7f0e')  # Horizontal bar chart for clarity
    plt.title(f'Top {n_features} Feature Importances', fontsize=14)
    plt.xlabel('Importance Score')
    plt.ylabel('Feature Name')
    plt.gca().invert_yaxis()  # Ensure the most important feature is at the top
    plt.tight_layout()
    importance_path = REPORT_DIR / f"{'combined' if 'mol_weight' in feature_names else 'fingerprint'}_top_{n_features}_feature_importance.png"
    plt.savefig(importance_path, dpi=300)
    plt.show()
    logger.info(f"Feature importance plot saved to: {importance_path}")


def run_fixed_rfr_model(X: np.ndarray, Y: np.ndarray, model_name: str, feature_names: Optional[list] = None):
    """
    Train a Random Forest Regressor (RFR) with a fixed, strong hyperparameter set and evaluate it.
    """
    logger.info(f"\n--- Starting training for {model_name} RFR model ---")
    
    # 1. Data split
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.3, random_state=42
    )

    # 2. Define model (strong defaults to avoid lengthy hyperparameter search)
    model = RandomForestRegressor(
        n_estimators=500,        # 500 trees
        max_depth=20,            # depth 20
        random_state=42,
        n_jobs=-1                
    )
    
    logger.info(f"Training {model_name}...")
    model.fit(X_train, Y_train)
    logger.info("Model training completed.")
    
    # 3. Model evaluation
    Y_pred = model.predict(X_test)
    mse = mean_squared_error(Y_test, Y_pred)
    r2 = r2_score(Y_test, Y_pred)
    
    logger.info(f"--- {model_name} Evaluation Results (Test Set) ---")
    logger.info(f"Mean Squared Error (MSE): {mse:.4f}")
    logger.info(f"R² Score: {r2:.4f}")
    
    # 4. Feature importance analysis
    if feature_names:
        analyze_feature_importance(model, feature_names)
        
    return model, r2

# =======================================================================
# III. Main (Entry point)
# =======================================================================
def main():
    """Main function: load data, prepare features, and run two analyses."""
    
    if not DATA_PATH.exists():
        logger.error(f"FATAL ERROR: Dataset file not found: {DATA_PATH.resolve()}")
        return

    try:
        df = pd.read_csv(DATA_PATH)
        df = df.dropna(subset=['standardized_smiles', 'pIC50'])
        logger.info(f"Data loaded. Valid records for modeling: {len(df)}")
        if len(df) == 0: return
        
        Y = df['pIC50'].values

        REPORT_DIR.mkdir(parents=True, exist_ok=True)

        # -----------------------------------------------------------
        # Analysis 1: Fingerprint-only (perform feature importance)
        # -----------------------------------------------------------
        X_fp = generate_fingerprint_features(df)
        fp_feature_names = [f"FP_Bit_{i}" for i in range(X_fp.shape[1])]
        
        fp_model, fp_r2 = run_fixed_rfr_model(X_fp, Y, "Fingerprint-Only", fp_feature_names)
        
        # -----------------------------------------------------------
        # Analysis 2: Fingerprint + numeric descriptors (feature fusion)
        # -----------------------------------------------------------
        X_combined, combined_feature_names = generate_combined_features(df)
        
        combined_model, combined_r2 = run_fixed_rfr_model(X_combined, Y, "Combined (FP + Descriptors)", combined_feature_names)

        # -----------------------------------------------------------
        # Compare results
        # -----------------------------------------------------------
        logger.info("\n--- Final model comparison ---")
        logger.info(f"Fingerprint-only R²: {fp_r2:.4f}")
        logger.info(f"Fingerprint + descriptors R²: {combined_r2:.4f}")

        if combined_r2 > fp_r2:
             logger.info("Conclusion: combining molecular descriptors slightly improved model performance.")
        else:
             logger.info("Conclusion: combining molecular descriptors did not produce a meaningful improvement.")
        
        logger.info("\n--- Traditional ML analysis stage completed ---")

    except Exception as e:
        logger.error(f"Unexpected error occurred in the model training pipeline: {e}", exc_info=True)


if __name__ == "__main__":
    main()
