import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
import warnings

# --- RDKit cheminformatics library ---
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit import DataStructs
except ImportError:
    # If RDKit import fails, display a clear error message
    print("FATAL ERROR: RDKit library not found or not properly installed. Please run 'conda install -c rdkit rdkit'")
    exit(1)

# --- Scikit-learn machine learning library ---
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import randint
import joblib

# --- Configuration ---
DATA_PATH = Path("data/processed/final_combined_dataset.csv")
REPORT_DIR = Path("reports/models")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore', category=UserWarning)

# =======================================================================
# I. Featurization
# =======================================================================

def smiles_to_ecfp4_fingerprint(smiles: str, radius: int = 2, nBits: int = 2048) -> np.ndarray:
    """Convert a SMILES string into an ECFP4 (Morgan Fingerprint) binary vector."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.zeros(nBits, dtype=int)
        
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=nBits)
        return np.array(fp, dtype=int)
    except Exception:
        return np.zeros(nBits, dtype=int)

def generate_fingerprint_features(df: pd.DataFrame, smiles_col: str = 'standardized_smiles', nBits: int = 2048) -> np.ndarray:
    """Generate molecular fingerprint feature matrix for the entire DataFrame."""
    logger.info(f"Starting conversion of {len(df)} molecules into {nBits}-bit ECFP4 fingerprints...")
    
    fingerprints = df[smiles_col].apply(
        lambda smi: smiles_to_ecfp4_fingerprint(smi, nBits=nBits)
    ).tolist()
    
    feature_matrix = np.array(fingerprints)
    logger.info(f"Feature matrix generation completed. Shape: {feature_matrix.shape}")
    
    return feature_matrix


# =======================================================================
# II. Model Training and Evaluation
# =======================================================================

def run_model_optimization(X: np.ndarray, Y: np.ndarray):
    """
    Perform data splitting, Random Forest hyperparameter tuning, training, and evaluation.
    """
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("--- Starting model optimization process (Randomized Search) ---")
    
    # 1. Split the dataset
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.3, random_state=42
    )
    logger.info(f"Training set size: {len(X_train)}, Test set size: {len(X_test)}")

    # 2. Define parameter space and model structure
    param_dist = {
        'n_estimators': randint(100, 1000),
        'max_depth': randint(10, 50),
        'min_samples_split': randint(2, 20),
        'min_samples_leaf': randint(1, 10)
    }

    base_model = RandomForestRegressor(random_state=42, n_jobs=-1)
    
    logger.info("Starting Randomized Search (30 iterations, 3-fold cross-validation)...")
    random_search = RandomizedSearchCV(
        base_model, 
        param_distributions=param_dist, 
        n_iter=30, 
        cv=3, 
        scoring='r2', 
        random_state=42, 
        n_jobs=-1,
        verbose=1
    )
    
    random_search.fit(X_train, Y_train)
    
    best_model = random_search.best_estimator_
    logger.info(f"Best RFR model parameters: {random_search.best_params_}")
    
    # 3. Model evaluation (using best model)
    Y_pred = best_model.predict(X_test)
    mse = mean_squared_error(Y_test, Y_pred)
    r2 = r2_score(Y_test, Y_pred)
    
    logger.info("--- Best model evaluation results (test set) ---")
    logger.info(f"Mean Squared Error (MSE): {mse:.4f}")
    logger.info(f"R² Score: {r2:.4f}")
    
    # Save the best model
    model_save_path = REPORT_DIR / "best_rfr_model.pkl"
    joblib.dump(best_model, model_save_path)
    logger.info(f"Best model saved to: {model_save_path}")

    # 4. Visualization of prediction results
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(8, 8))
    sns.scatterplot(x=Y_test, y=Y_pred, alpha=0.6, color='#1f77b4')
    min_val = min(Y_test.min(), Y_pred.min())
    max_val = max(Y_test.max(), Y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect prediction line')
    plt.xlabel("True pIC50 value", fontsize=12)
    plt.ylabel("Predicted pIC50 value", fontsize=12)
    plt.title(f"RFR Best Model Predictions (R² = {r2:.3f})", fontsize=14, fontweight='bold')
    plt.legend()
    report_path = REPORT_DIR / "RFR_best_prediction_vs_actual.png"
    plt.savefig(report_path, dpi=300, bbox_inches='tight')
    plt.show()
    logger.info(f"Best model evaluation plot generated and saved to: {report_path}")

    # 5. Feature Importance Analysis
    analyze_feature_importance(best_model, n_features=100)
    
    return best_model 


def analyze_feature_importance(model: RandomForestRegressor, n_features: int = 100):
    """
    Extract feature importances from the trained model and display the top N bits.
    """
    if not hasattr(model, 'feature_importances_'):
        logger.warning("Model does not have 'feature_importances_' attribute. Skipping feature importance analysis.")
        return

    logger.info(f"\n--- Feature Importance Analysis (Top {n_features} ECFP4 bits) ---")
    
    # 1. Retrieve all feature importances
    importances = pd.Series(model.feature_importances_)
    
    # 2. Select the top N contributing bits
    top_N_importances = importances.nlargest(n_features)
    
    # 3. Count the number of non-zero importance bits
    non_zero_count = (importances > 0).sum()
    
    logger.info(f"There are 2048 fingerprint bits in total, of which {non_zero_count} bits have importance > 0.")
    logger.info(f"Top {n_features} most important fingerprint bits and their importances:\n" + top_N_importances.to_string())
    
    # Visualize top N important features
    plt.figure(figsize=(10, 6))
    top_N_importances.plot(kind='bar', color='#ff7f0e')
    plt.title(f'Top {n_features} ECFP4 Fingerprint Feature Importances')
    plt.xlabel('ECFP4 Bit Index')
    plt.ylabel('Importance Score')
    plt.tight_layout()
    importance_path = REPORT_DIR / f"top_{n_features}_feature_importance.png"
    plt.savefig(importance_path, dpi=300)
    plt.show()
    logger.info(f"Feature importance plot saved to: {importance_path}")


# =======================================================================
# III. Main
# =======================================================================
def main():
    """Main function: load data, prepare features, and train the model."""
    
    # --- Critical check: file path ---
    if not DATA_PATH.exists():
        logger.error(f"FATAL ERROR: Dataset file not found: {DATA_PATH.resolve()}")
        logger.error("Please verify that the file path is correct or that data_pipeline.py has been run successfully.")
        return

    try:
        # 1. Load data
        logger.info("Loading dataset...")
        df = pd.read_csv(DATA_PATH)
        
        # Preprocessing: remove rows with missing SMILES or pIC50 values
        df = df.dropna(subset=['standardized_smiles', 'pIC50'])
        logger.info(f"Data loaded. Valid records for modeling: {len(df)}")
        
        if len(df) == 0:
            logger.error("No valid compounds or activity values found in the dataset. Please check the source data.")
            return
        
        # 2. Define target (Y) and features (X)
        Y = df['pIC50'].values
        
        # Use only molecular fingerprints (ECFP4, 2048-bit)
        X = generate_fingerprint_features(df)
        
        # 3. Run model training and evaluation
        run_model_optimization(X, Y)
        
    except Exception as e:
        logger.error(f"Unexpected error occurred during the model training pipeline: {e}", exc_info=True)


if __name__ == "__main__":
    main()
