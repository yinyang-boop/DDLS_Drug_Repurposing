import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
import warnings

# --- Explicitly import RDKit modules ---
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit import DataStructs
except ImportError:
    # If RDKit import fails, show a clear error message
    print("FATAL ERROR: RDKit library not found or not properly installed. Please run 'conda install -c rdkit rdkit'")
    exit(1)

# --- Scikit-learn machine learning library ---
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# --- Configuration ---
# Ensure file paths are correct relative to the project root
DATA_PATH = Path("data/processed/final_combined_dataset.csv")
REPORT_DIR = Path("reports/models")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore', category=UserWarning)

# =======================================================================
# I. Molecular Featurization
# =======================================================================

def smiles_to_ecfp4_fingerprint(smiles: str, radius: int = 2, nBits: int = 2048) -> np.ndarray:
    """
    Convert a SMILES string into an ECFP4 (Morgan Fingerprint) binary vector.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.zeros(nBits, dtype=int)
        
        # Morgan Fingerprint with radius=2 (ECFP4)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=nBits)
        
        return np.array(fp, dtype=int)
    except Exception as e:
        # Log minor fingerprint generation errors without interrupting the main process
        logger.debug(f"SMILES fingerprint generation failed ({smiles}): {e}")
        return np.zeros(nBits, dtype=int)


def generate_fingerprint_features(df: pd.DataFrame, smiles_col: str = 'standardized_smiles', nBits: int = 2048) -> np.ndarray:
    """
    Generate molecular fingerprint feature matrix for the entire DataFrame.
    """
    logger.info(f"Starting conversion of {len(df)} molecules into {nBits}-bit ECFP4 fingerprints...")
    
    fingerprints = df[smiles_col].apply(
        lambda smi: smiles_to_ecfp4_fingerprint(smi, nBits=nBits)
    ).tolist()
    
    feature_matrix = np.array(fingerprints)
    logger.info(f"Feature matrix generation completed, shape: {feature_matrix.shape}")
    
    return feature_matrix


def run_model_training(X: np.ndarray, Y: np.ndarray):
    """
    Execute data splitting, Random Forest regression model training, and evaluation.
    """
    REPORT_DIR.mkdir(parents=True, exist_ok=True)  # Ensure the report directory exists
    logger.info("--- Starting model training pipeline ---")
    
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.3, random_state=42
    )
    logger.info(f"Training set size: {len(X_train)}, Test set size: {len(X_test)}")

    model = RandomForestRegressor(
        n_estimators=500,        
        max_depth=20,            
        random_state=42,
        n_jobs=-1                
    )
    
    logger.info("Training Random Forest model...")
    model.fit(X_train, Y_train)
    logger.info("Model training completed.")
    
    Y_pred = model.predict(X_test)
    
    mse = mean_squared_error(Y_test, Y_pred)
    r2 = r2_score(Y_test, Y_pred)
    
    logger.info("--- Model Evaluation Results (Test Set) ---")
    logger.info(f"Mean Squared Error (MSE): {mse:.4f}")
    logger.info(f"R² Score: {r2:.4f}")
    
    # 4. Visualization: Predicted vs. Actual Values
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(8, 8))
    
    sns.scatterplot(x=Y_test, y=Y_pred, alpha=0.6, color='#1f77b4')
    
    min_val = min(Y_test.min(), Y_pred.min())
    max_val = max(Y_test.max(), Y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction Line')
    
    plt.xlabel("Actual pIC50 Values", fontsize=12)
    plt.ylabel("Predicted pIC50 Values", fontsize=12)
    plt.title(f"Random Forest Regression (R² = {r2:.3f})", fontsize=14, fontweight='bold')
    plt.legend()
    
    report_path = REPORT_DIR / "RFR_prediction_vs_actual.png"
    plt.savefig(report_path, dpi=300, bbox_inches='tight')
    plt.show()

    logger.info(f"Model evaluation plot generated and saved to: {report_path}")

# =======================================================================
# III. Main Pipeline
# =======================================================================
def main():
    """Main function: Load data, prepare features, and train the model"""
    
    # --- Critical check: file path ---
    if not DATA_PATH.exists():
        logger.error(f"FATAL ERROR: Dataset file not found: {DATA_PATH.resolve()}")
        logger.error("Please verify the path or check if data_pipeline.py has run successfully.")
        return

    try:
        # 1. Load dataset
        logger.info("Loading dataset...")
        df = pd.read_csv(DATA_PATH)
        
        # Preprocess: remove rows with missing SMILES or pIC50 values
        df = df.dropna(subset=['standardized_smiles', 'pIC50'])
        logger.info(f"Dataset loaded. Valid records for modeling: {len(df)}")
        
        if len(df) == 0:
            logger.error("No valid compounds or activity values found in the dataset. Please check the source data.")
            return
        
        # 2. Define target (Y) and features (X)
        Y = df['pIC50'].values
        
        # Use molecular fingerprints only (ECFP4, 2048-bit)
        X = generate_fingerprint_features(df)
        
        # 3. Run model training and evaluation
        run_model_training(X, Y)
        
    except Exception as e:
        logger.error(f"Unexpected error occurred during model training pipeline: {e}", exc_info=True)


if __name__ == "__main__":
    main()
