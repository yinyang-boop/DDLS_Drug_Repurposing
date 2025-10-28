import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import logging
from pathlib import Path

# --- Configuration ---
DATA_PATH = Path("data/processed/final_combined_dataset.csv")
REPORT_DIR = Path("reports/analysis")
# Ensure the report directory exists
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# Configure logging (consistent with the pipeline)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Ensure PandasTools warnings do not affect this script's output
import warnings
warnings.filterwarnings('ignore')

def load_data():
    """Load the final consolidated dataset"""
    if not DATA_PATH.exists():
        logger.error(f"Dataset file not found: {DATA_PATH}. Please ensure data_pipeline.py ran successfully.")
        return None
    
    logger.info(f"Loading dataset: {DATA_PATH}")
    try:
        df = pd.read_csv(DATA_PATH)
        logger.info(f"Dataset loaded successfully, total records: {len(df)}")
        return df
    except Exception as e:
        logger.error(f"Error occurred while loading CSV file: {e}")
        return None

def analyze_distributions(df: pd.DataFrame):
    """
    Inspect distributions of key properties (pIC50, MW, LogP, TPSA).
    """
    logger.info("--- 1. Key property distribution analysis ---")
    
    # Define descriptor columns to plot
    # Note: pIC50 is the activity value; others are descriptors
    descriptor_cols = ['pIC50', 'mol_weight', 'logp', 'tpsa', 'hbd', 'hba']
    
    # Keep only columns that exist in the dataframe
    plot_cols = [col for col in descriptor_cols if col in df.columns]
    
    if not plot_cols:
        logger.warning("The dataset is missing key descriptor columns required for analysis.")
        return

    # Plot distributions (histogram + KDE)
    fig, axes = plt.subplots(nrows=(len(plot_cols) // 3) + 1, ncols=3, figsize=(18, 5 * ((len(plot_cols) // 3) + 1)))
    axes = axes.flatten()  # Flatten the 2D axes array for easy indexing

    for i, col in enumerate(plot_cols):
        # Use dropna() to plot only valid values
        sns.histplot(df[col].dropna(), kde=True, ax=axes[i], color='#1f77b4')
        axes[i].set_title(f'Distribution: {col}')
        axes[i].set_xlabel(col)
        axes[i].grid(axis='y', alpha=0.5)

    # Remove unused subplots
    for j in range(len(plot_cols), len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])  # Adjust layout to fit title
    plt.suptitle("Key Molecular Properties and Activity Distributions", fontsize=16, fontweight='bold')
    
    report_path = REPORT_DIR / "1_descriptor_distributions.png"
    plt.savefig(report_path, dpi=300, bbox_inches='tight')
    plt.show()
    logger.info(f"Distribution plots saved to: {report_path}")


def analyze_correlations(df: pd.DataFrame):
    """
    Verify correlations between key descriptors, especially with activity (pIC50).
    """
    logger.info("--- 2. Key descriptor correlation analysis ---")
    
    # Columns to use for correlation analysis
    # Must include activity pIC50 and all numeric descriptors
    corr_cols = ['pIC50', 'mol_weight', 'logp', 'hbd', 'hba', 'tpsa', 'heavy_atoms', 'rotatable_bonds']
    corr_cols = [col for col in corr_cols if col in df.columns]
    
    if 'pIC50' not in corr_cols:
        logger.error("The dataset is missing the 'pIC50' column; cannot perform activity correlation analysis.")
        return

    # 1. Compute correlation matrix
    correlation_matrix = df[corr_cols].corr()
    
    # 2. Save correlation values
    corr_path = REPORT_DIR / "2_correlation_values.csv"
    correlation_matrix.to_csv(corr_path)
    logger.info(f"Correlation matrix saved to: {corr_path}")
    
    # 3. Plot heatmap
    plt.figure(figsize=(10, 8))
    # annot=True displays values; fmt=".2f" keeps two decimal places
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', 
                linewidths=.5, linecolor='black')
    plt.title('Correlation Heatmap: Key Molecular Descriptors vs Activity (pIC50)')
    
    corr_plot_path = REPORT_DIR / "2_correlation_heatmap.png"
    plt.savefig(corr_plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    logger.info(f"Correlation heatmap saved to: {corr_plot_path}")
    
    # 4. Highlight correlations with pIC50
    pIC50_corr = correlation_matrix['pIC50'].sort_values(ascending=False)
    logger.info("\n--- Correlation ranking: pIC50 vs descriptors ---\n" + pIC50_corr.to_string())


def main():
    df = load_data()
    
    if df is not None:
        # Perform distribution analysis
        analyze_distributions(df)
        
        # Perform correlation analysis
        analyze_correlations(df)
        
        logger.info("Distribution and correlation analysis completed.")

if __name__ == "__main__":
    main()
