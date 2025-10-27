# mcp_tools.py - (MCP Toolset)

import json
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
from rdkit.DataStructs import FingerprintSimilarity
from typing import Dict, Any, List

class MolecularAnalyzer:

    @staticmethod
    def calculate_descriptors(smiles: str) -> Dict[str, float]:
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            return {}

        return {
            'MW (g/mol)': Descriptors.MolWt(mol),
            'LogP': Descriptors.MolLogP(mol),
            'H Donors': Descriptors.NumHDonors(mol),
            'H Acceptors': Descriptors.NumHAcceptors(mol)
        }

    @staticmethod
    def calculate_tanimoto_similarity(smiles1: str, smiles2: str) -> float:
        mol1 = Chem.MolFromSmiles(smiles1)
        mol2 = Chem.MolFromSmiles(smiles2)

        if not (mol1 and mol2):
            return 0.0

        fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, radius=2, nBits=2048)
        fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, radius=2, nBits=2048)
        
        return FingerprintSimilarity(fp1, fp2)

class DeepDTAProcessor:

    def __init__(self, encoding_file_path: str = 'deepdta_encoding.json'):
        try:
            with open(encoding_file_path, 'r') as f:
                self.encoding_data = json.load(f)
            self.smiles_dict: Dict[str, int] = self.encoding_data['smiles_dict']
            self.protein_dict: Dict[str, int] = self.encoding_data['protein_dict']
        except FileNotFoundError:
            print(f"Error: Encoding file not found at {encoding_file_path}")
            self.smiles_dict = {}
            self.protein_dict = {}

    def encode_smiles(self, smiles: str, max_len: int = 100) -> List[int]:
        if not self.smiles_dict:
            return []
        
        encoded = [self.smiles_dict.get(char, 0) for char in smiles]
        
        if len(encoded) > max_len:
            return encoded[:max_len]
        elif len(encoded) < max_len:
            return encoded + [0] * (max_len - len(encoded))
        return encoded

    def encode_protein(self, sequence: str, max_len: int = 1000) -> List[int]:

        if not self.protein_dict:
            return []
            
        encoded = [self.protein_dict.get(char, 0) for char in sequence]

        if len(encoded) > max_len:
            return encoded[:max_len]
        elif len(encoded) < max_len:
            return encoded + [0] * (max_len - len(encoded))
        return encoded

    @staticmethod
    def clean_dta_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        
        # 1. delete any rows with missing values（SMILES, Sequence, pIC50）
        required_cols = ['canonical_smiles', 'target_sequence', 'pIC50']
        df_cleaned = df.dropna(subset=required_cols).copy()
        
        # 2. maintain pIC50 float
        df_cleaned['pIC50'] = pd.to_numeric(df_cleaned['pIC50'], errors='coerce')
        df_cleaned = df_cleaned.dropna(subset=['pIC50'])
        
        # 3. (consider about using RDKit to validate SMILES effectiveness)
        
        return df_cleaned


# --- Sample Use (Test Only) ---
if __name__ == '__main__':
    
    print("--- 1. MolecularAnalyzer Test ---")
    smiles_t = "CCCCCCCCCCCCCCCCCCCC(=O)NCCc1cc(N)c(C(=O)O)cc1"
    smiles_s = "CCCCCCCCCCCCCCCCCC(=O)NCCc1ccc(C(=O)O)cc1"
    
    desc_t = MolecularAnalyzer.calculate_descriptors(smiles_t)
    similarity = MolecularAnalyzer.calculate_tanimoto_similarity(smiles_t, smiles_s)
    
    print("Tirzepatide Descriptors:", desc_t)
    print(f"Tanimoto Similarity: {similarity:.4f}")

    print("\n--- 2. DeepDTAProcessor Test ---")
    try:
        dta_processor = DeepDTAProcessor(encoding_file_path='deepdta_encoding.json')
        
        test_smiles = "CC(=O)Oc1ccccc1C(=O)O" # Aspirin
        encoded_smiles = dta_processor.encode_smiles(test_smiles, max_len=50)
        
        print(f"Test SMILES: {test_smiles}")
        print(f"Encoded Length: {len(encoded_smiles)}")
        print(f"Encoded Snippet: {encoded_smiles[:10]}...")
    
        df = pd.read_csv('deepdta_dataset.csv')
        df_cleaned = DeepDTAProcessor.clean_dta_dataframe(df.head(100))
        print(f"Original shape: {df.head(100).shape}, Cleaned shape: {df_cleaned.shape}")

    except Exception as e:
        print(f"DeepDTAProcessor Test Failed (Ensure files are present): {e}")
