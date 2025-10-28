#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tirzepatide and Semaglutide simplified structure analysis and visualization script.
Uses RDKit to compute molecular properties and Matplotlib/Seaborn for comparative visualization.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import AllChem 
from rdkit.DataStructs import FingerprintSimilarity
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define color theme
COLORS = {
    'Semaglutide': '#1f77b4', # Blue
    'Tirzepatide': '#ff7f0e', # Orange
    'Highlight': '#2ca02c'    # Green
}

# Simplified drug structure data
DRUG_DATA = {
    "Semaglutide": "CCCCCCCCCCCCCCCCCC(=O)NCCc1ccc(C(=O)O)cc1",
    "Tirzepatide": "CCCCCCCCCCCCCCCCCCCC(=O)NCCc1cc(N)c(C(=O)O)cc1"
}

def calculate_descriptors(smiles):
    """Compute molecular descriptors for a single SMILES string."""
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        logger.error(f"Cannot parse SMILES: {smiles}")
        return None

    # Compute key descriptors
    descriptors = {
        'MW (g/mol)': Descriptors.MolWt(mol),
        'LogP': Descriptors.MolLogP(mol),
        'H Donors': Descriptors.NumHDonors(mol),
        'H Acceptors': Descriptors.NumHAcceptors(mol)
    }
    return descriptors

def create_comparison_dataframe():
    """Compute descriptors for all drugs and return a DataFrame."""
    data_list = []
    for name, smiles in DRUG_DATA.items():
        desc = calculate_descriptors(smiles)
        if desc:
            row = {'Drug Name': name, 'SMILES': smiles, **desc}
            data_list.append(row)
    
    df = pd.DataFrame(data_list)
    df.set_index('Drug Name', inplace=True)
    return df

def visualize_descriptors(df: pd.DataFrame, save_path: Path):
    """Generate a bar plot comparing physicochemical properties."""
    
    # Convert from wide to long format for Seaborn plotting
    df_plot = df.drop(columns=['SMILES']).reset_index().melt(
        id_vars='Drug Name', 
        var_name='Property', 
        value_name='Value'
    )

    plt.figure(figsize=(14, 7))
    sns.barplot(
        x='Property', 
        y='Value', 
        hue='Drug Name', 
        data=df_plot, 
        palette=[COLORS['Semaglutide'], COLORS['Tirzepatide']]
    )

    # Add value labels
    for container in plt.gca().containers:
        plt.bar_label(container, fmt='%.1f', fontsize=10)

    # Note: Original Chinese title may not render properly depending on environment font
    plt.title('Tirzepatide vs Semaglutide Simplified Structure Physicochemical Comparison', fontsize=16, fontweight='bold')
    plt.xlabel('Molecular Property', fontsize=12)
    plt.ylabel('Value', fontsize=12)
    plt.legend(title='Drug', fontsize=10, title_fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(save_path / 'physicochemical_comparison.png', dpi=300)
    logger.info(f"Physicochemical comparison plot saved at: {save_path / 'physicochemical_comparison.png'}")
    plt.close()

def visualize_similarity(df: pd.DataFrame, save_path: Path):
    """Compute and visualize Morgan fingerprint similarity."""
    smiles1 = df.loc['Semaglutide', 'SMILES']
    smiles2 = df.loc['Tirzepatide', 'SMILES']
    
    mol1 = Chem.MolFromSmiles(smiles1)
    mol2 = Chem.MolFromSmiles(smiles2)
    
    if not (mol1 and mol2):
        logger.error("Cannot compute similarity, molecule parsing failed.")
        return

    # Compute Morgan fingerprints as bit vectors
    fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2, nBits=2048)
    fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2, nBits=2048)

    # Tanimoto similarity
    similarity = FingerprintSimilarity(fp1, fp2)
    
    # Simple visualization as horizontal bar
    plt.figure(figsize=(8, 4))
    
    plt.barh(['Tanimoto Similarity'], [similarity], color=COLORS['Highlight'])
    plt.xlim(0, 1.0)
    plt.title('Semaglutide vs Tirzepatide (Simplified Structure) Structural Similarity', fontsize=14, fontweight='bold')
    plt.xlabel('Tanimoto Similarity (0.0 - 1.0)', fontsize=12)
    
    # Add value label
    plt.text(similarity, 0, f'{similarity:.3f}', va='center', ha='right' if similarity < 0.9 else 'left', 
             color='white' if similarity < 0.9 else COLORS['Highlight'], 
             fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path / 'structural_similarity.png', dpi=300)
    logger.info(f"Structural similarity plot saved at: {save_path / 'structural_similarity.png'}")
    plt.close()


def main():
    """Main execution function."""
    logger.info("--- Starting drug structure analysis and visualization ---")
    
    # Create output directory for reports/figures
    output_dir = Path('./reports/figures')
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory confirmed: {output_dir}")

    # 1. Prepare data and compute descriptors
    comparison_df = create_comparison_dataframe()
    if comparison_df.empty:
        logger.error("No analysis data available, cannot proceed.")
        return
        
    print("\n--- Simplified Drug Structure Physicochemical Summary ---\n")
    print(comparison_df.to_markdown())
    print("\n----------------------------------\n")

    # 2. Generate visualizations
    visualize_descriptors(comparison_df, output_dir)
    visualize_similarity(comparison_df, output_dir)
    
    logger.info("\n--- Visualization analysis completed ---")
    print(f"Generated image files can be found in {output_dir}:")
    print(" - physicochemical_comparison.png (physicochemical comparison)")
    print(" - structural_similarity.png (structural similarity)")
    
    print("\n>>> Visualization insight: Tirzepatide (C20 side chain) exhibits higher molecular weight and LogP compared to Semaglutide (C18 side chain), potentially conferring better membrane permeability and CFTR lipid environment binding. Additionally, Tirzepatide has one extra H-bond donor/acceptor (in simplified structure) which may provide additional anchoring points for CFTR binding.")

if __name__ == '__main__':
    main()
