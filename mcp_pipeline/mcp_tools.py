# mcp_tools.py
# Extended MCP Toolset with molecular and sequence analysis utilities

import json
import pandas as pd
import numpy as np
from typing import Dict, List
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit.Chem import rdFingerprintGenerator
from rdkit.DataStructs import TanimotoSimilarity, BulkTanimotoSimilarity

# Optional: PubChem resolver
try:
    import pubchempy as pcp
    HAS_PUBCHEMPY = True
except ImportError:
    HAS_PUBCHEMPY = False


class MolecularAnalyzer:
    """Tools for molecular descriptor calculation and SMILES-based similarity."""

    @staticmethod
    def resolve_smiles(name_or_cid: str) -> str:
        """Resolve a compound name or CID to canonical SMILES using PubChem."""
        if HAS_PUBCHEMPY:
            try:
                if isinstance(name_or_cid, int) or str(name_or_cid).isdigit():
                    comp = pcp.Compound.from_cid(int(name_or_cid))
                else:
                    comp = pcp.get_compounds(name_or_cid, "name")[0]
                return comp.connectivity_smiles
            except Exception:
                return ""
        return ""

    @staticmethod
    def calculate_descriptors(smiles: str) -> Dict[str, float]:
        """Calculate common molecular descriptors from a SMILES string."""
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            return {}
        return {
            "MW (g/mol)": Descriptors.MolWt(mol),
            "LogP": Descriptors.MolLogP(mol),
            "H Donors": Descriptors.NumHDonors(mol),
            "H Acceptors": Descriptors.NumHAcceptors(mol),
            "TPSA": rdMolDescriptors.CalcTPSA(mol),
            "Rotatable Bonds": Descriptors.NumRotatableBonds(mol),
            "Heavy Atoms": Descriptors.HeavyAtomCount(mol),
        }

    @staticmethod
    def calculate_tanimoto_similarity(smiles1: str, smiles2: str) -> float:
        """Calculate Tanimoto similarity between two molecules."""
        mol1, mol2 = Chem.MolFromSmiles(smiles1), Chem.MolFromSmiles(smiles2)
        if not (mol1 and mol2):
            return 0.0
        gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
        fp1 = gen.GetFingerprint(mol1)
        fp2 = gen.GetFingerprint(mol2)
        return TanimotoSimilarity(fp1, fp2)

    @staticmethod
    def similarity_matrix(smiles_list: List[str]) -> pd.DataFrame:
        """Generate a similarity matrix for a list of SMILES strings."""
        mols = [Chem.MolFromSmiles(s) for s in smiles_list if Chem.MolFromSmiles(s)]
        gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
        fps = [gen.GetFingerprint(m) for m in mols]
        n = len(fps)
        sim_matrix = np.zeros((n, n))
        for i in range(n):
            sim_matrix[i, :] = BulkTanimotoSimilarity(fps[i], fps)
        return pd.DataFrame(sim_matrix, index=smiles_list[:n], columns=smiles_list[:n])


class SequenceAnalyzer:
    """Tools for peptide/protein sequence similarity."""

    @staticmethod
    def sequence_identity(seq1: str, seq2: str) -> float:
        """Compute simple sequence identity (fraction of matching positions)."""
        if not seq1 or not seq2:
            return 0.0
        min_len = min(len(seq1), len(seq2))
        matches = sum(1 for a, b in zip(seq1[:min_len], seq2[:min_len]) if a == b)
        return matches / max(len(seq1), len(seq2))

    @staticmethod
    def levenshtein_distance(seq1: str, seq2: str) -> int:
        """Compute Levenshtein edit distance between two sequences."""
        if len(seq1) < len(seq2):
            return SequenceAnalyzer.levenshtein_distance(seq2, seq1)
        if len(seq2) == 0:
            return len(seq1)
        previous_row = range(len(seq2) + 1)
        for i, c1 in enumerate(seq1):
            current_row = [i + 1]
            for j, c2 in enumerate(seq2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        return previous_row[-1]

    @staticmethod
    def normalized_similarity(seq1: str, seq2: str) -> float:
        """Convert Levenshtein distance to normalized similarity (0–1)."""
        if not seq1 or not seq2:
            return 0.0
        dist = SequenceAnalyzer.levenshtein_distance(seq1, seq2)
        max_len = max(len(seq1), len(seq2))
        return 1 - dist / max_len


class DeepDTAProcessor:
    """Encoding and dataset utilities for DeepDTA."""

    def __init__(self, encoding_file_path: str = "deepdta_encoding.json"):
        try:
            with open(encoding_file_path, "r") as f:
                self.encoding_data = json.load(f)
            self.smiles_dict: Dict[str, int] = self.encoding_data.get("smiles_dict", {})
            self.protein_dict: Dict[str, int] = self.encoding_data.get("protein_dict", {})
        except FileNotFoundError:
            print(f"Error: Encoding file not found at {encoding_file_path}")
            self.smiles_dict = {}
            self.protein_dict = {}

    def encode_smiles(self, smiles: str, max_len: int = 100) -> List[int]:
        """Encode a SMILES string into integer indices."""
        if not self.smiles_dict:
            return []
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            return []
        encoded = [self.smiles_dict.get(ch, 0) for ch in smiles]
        return encoded[:max_len] + [0] * max(0, max_len - len(encoded))

    def encode_protein(self, sequence: str, max_len: int = 1000) -> List[int]:
        """Encode a protein sequence into integer indices."""
        if not self.protein_dict:
            return []
        encoded = [self.protein_dict.get(ch, 0) for ch in sequence]
        return encoded[:max_len] + [0] * max(0, max_len - len(encoded))

    @staticmethod
    def clean_dta_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """Clean a DeepDTA dataset DataFrame."""
        required_cols = ["canonical_smiles", "target_sequence", "pIC50"]
        df_cleaned = df.dropna(subset=required_cols).copy()
        df_cleaned["pIC50"] = pd.to_numeric(df_cleaned["pIC50"], errors="coerce")
        df_cleaned = df_cleaned.dropna(subset=["pIC50"])
        return df_cleaned

    @staticmethod
    def split_dataset(df: pd.DataFrame, train_frac=0.8, val_frac=0.1, seed=42):
        """Split dataset into train/validation/test sets."""
        df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
        n = len(df)
        train_end = int(train_frac * n)
        val_end = int((train_frac + val_frac) * n)
        return df[:train_end], df[train_end:val_end], df[val_end:]
