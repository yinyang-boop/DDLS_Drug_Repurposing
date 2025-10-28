import pandas as pd
import numpy as np
import requests
import re
import json
from pathlib import Path
import logging
import warnings
from urllib.parse import urljoin  # Used for safe URL concatenation

# --- Configuration ---
# EGFR ChEMBL ID and UniProt ID
TARGET_CHEMBL_ID = 'CHEMBL203'
TARGET_UNIPROT_ID = 'P00533'
DATA_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
ENCODING_PATH = PROCESSED_DIR / "deepdta_encoding.json"
FINAL_DATA_PATH = PROCESSED_DIR / "deepdta_dataset.csv"

# ChEMBL API configuration: use base API path
CHEMBL_API_BASE = "https://www.ebi.ac.uk/chembl/api/data"
UNIPROT_API_URL = "https://rest.uniprot.org/uniprotkb"

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore', category=UserWarning)

# =======================================================================
# I. Sequence Retrieval and Data Collection
# =======================================================================

def fetch_protein_sequence(uniprot_id: str) -> str:
    """Retrieve the full protein sequence for a given UniProt ID."""
    logger.info(f"Fetching sequence for target {uniprot_id} from UniProt API...")
    
    url = f"{UNIPROT_API_URL}/{uniprot_id}?format=json&fields=sequence"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        sequence = data.get('sequence', {}).get('value', '')
        
        if sequence:
            logger.info(f"Sequence retrieved successfully. Length: {len(sequence)}")
            return sequence
        else:
            logger.error(f"No sequence found in UniProt response. Data: {data}")
            return ""
            
    except requests.exceptions.RequestException as e:
        logger.error(f"UniProt API request failed: {e}")
        return ""

def calculate_pIC50(row) -> float:
    """Convert activity values with different units into pIC50."""
    unit = row['standard_units']
    value = row['standard_value']
    
    # Convert to Molar (M)
    if unit == 'nM':
        M_value = value * 1e-9
    elif unit in ('uM', 'µM'):
        M_value = value * 1e-6
    elif unit == 'mM':
        M_value = value * 1e-3
    elif unit == 'M':
        M_value = value
    else:
        return np.nan  # Unrecognized or unconvertible unit

    # Exclude non-positive values to avoid log(0)
    if M_value <= 0:
        return np.nan

    return -np.log10(M_value)


def fetch_and_process_chembl_data(target_chembl_id: str) -> pd.DataFrame:
    """Fetch raw IC50 data from ChEMBL API and compute pIC50."""
    logger.info(f"Fetching raw IC50 activity data for target {target_chembl_id} from ChEMBL API...")
    
    ACTIVITY_ENDPOINT = f"{CHEMBL_API_BASE}/activity.json"
    
    # Key fix: filter only for standard_type=IC50
    params = {
        'target_chembl_id': target_chembl_id,
        'standard_type': 'IC50',
        'limit': 1000
    }
    
    all_activities = []
    response = None
    
    # Initial request using parameter dictionary
    try:
        response = requests.get(ACTIVITY_ENDPOINT, 
                                params=params, 
                                headers={'Accept': 'application/json'}, 
                                timeout=30)
        logger.info(f"Requesting URL: {response.url}")
    except requests.exceptions.RequestException as e:
        logger.error(f"Initial ChEMBL API request failed: {e}") 
        return pd.DataFrame()

    # Handle pagination
    while True:
        try:
            response.raise_for_status() 
            data = response.json()
            
            if 'activities' in data and data['activities']:
                all_activities.extend(data['activities'])
            else:
                logger.warning("Current request returned 0 activity records or reached the last page.")
                break 
            
            # Handle pagination
            meta = data.get('page_meta', {})
            if 'next' in meta and meta['next'] is not None:
                next_relative_path = meta['next']
                
                # Use urljoin for safe URL concatenation
                query_url = urljoin(f"{CHEMBL_API_BASE}/", next_relative_path)
                
                logger.info(f"Fetching next page: {query_url}")
                # Fetch the next page
                response = requests.get(query_url, headers={'Accept': 'application/json'}, timeout=30)
            else:
                break  # No more pages
                
        except requests.exceptions.RequestException as e:
            logger.error(f"ChEMBL API request failed: {e}. Last URL: {response.url}") 
            break
        except Exception as e:
            logger.error(f"Unexpected error while processing ChEMBL response: {e}. Last URL: {response.url}")
            break

    if not all_activities:
        logger.error("No valid activity data retrieved.")
        return pd.DataFrame()

    logger.info(f"Successfully retrieved {len(all_activities)} raw IC50 records.")
    df = pd.json_normalize(all_activities)

    # 1. Filter and clean data
    df = df[['canonical_smiles', 'standard_units', 'standard_value', 'molecule_chembl_id']]
    
    # Drop rows with missing SMILES or values
    df = df.dropna(subset=['canonical_smiles', 'standard_value'])
    df['standard_value'] = pd.to_numeric(df['standard_value'], errors='coerce')
    df = df.dropna(subset=['standard_value'])

    # 2. Calculate pIC50
    df['pIC50'] = df.apply(calculate_pIC50, axis=1)
    df = df.dropna(subset=['pIC50'])
    
    # 3. Remove duplicate compounds (keep the highest activity, i.e., highest pIC50)
    df = df.sort_values(by='pIC50', ascending=False)
    df = df.drop_duplicates(subset=['canonical_smiles'], keep='first')
    
    df = df[['molecule_chembl_id', 'canonical_smiles', 'pIC50']]
    
    logger.info(f"After cleaning and pIC50 calculation, valid records: {len(df)}")
    return df

# =======================================================================
# II. DeepDTA Feature Encoding
# =======================================================================

def convert_numpy_types(obj):
    """Recursively convert NumPy/Pandas types into native Python types."""
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(elem) for elem in obj]
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()  # Convert np.int64 or np.float64 to standard int/float
    elif isinstance(obj, pd.Series):
        return obj.apply(convert_numpy_types).tolist()
    else:
        return obj

def create_char_dict(chars: list) -> dict:
    """Create a character-to-integer encoding dictionary."""
    # 0 is reserved for padding
    return {char: i + 1 for i, char in enumerate(chars)}

def build_deepdta_encoding(df: pd.DataFrame, protein_sequence: str):
    """
    Build character encoding dictionaries for SMILES and protein sequences.
    """
    # 1. SMILES character set
    smiles_chars = set()
    for smiles in df['canonical_smiles'].astype(str):
        smiles_chars.update(list(smiles)) 
        
    smiles_char_list = sorted(list(smiles_chars))
    logger.info(f"Number of unique SMILES characters: {len(smiles_char_list)}")
    
    # 2. Protein amino acid character set
    protein_chars = set(list(protein_sequence))
    protein_char_list = sorted(list(protein_chars))
    logger.info(f"Number of unique protein sequence characters: {len(protein_char_list)}")
    
    # 3. Create dictionaries
    smiles_dict = create_char_dict(smiles_char_list)
    protein_dict = create_char_dict(protein_char_list)

    # 4. Store encodings
    encoding_data = {
        "smiles_chars": smiles_char_list,
        "protein_chars": protein_char_list,
        "smiles_dict": smiles_dict,
        "protein_dict": protein_dict,
        "max_smiles_len": df['canonical_smiles'].str.len().max(),
        "max_protein_len": len(protein_sequence)
    }
    
    # Ensure output directory exists
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    
    # --- Key fix: Convert all NumPy/Pandas types before dumping ---
    serializable_encoding_data = convert_numpy_types(encoding_data)

    with open(ENCODING_PATH, 'w') as f:
        json.dump(serializable_encoding_data, f, indent=4)
        
    logger.info(f"DeepDTA encoding dictionary saved to: {ENCODING_PATH}")
    return encoding_data

# =======================================================================
# III. Main Workflow
# =======================================================================
def main():
    """Execute the DeepDTA data preparation pipeline."""
    
    # 1. Retrieve target sequence
    protein_sequence = fetch_protein_sequence(TARGET_UNIPROT_ID)
    if not protein_sequence:
        logger.error("Failed to retrieve target sequence. DeepDTA data preparation aborted.")
        return

    # 2. Retrieve activity data
    df = fetch_and_process_chembl_data(TARGET_CHEMBL_ID)
    if df.empty:
        logger.error("Failed to retrieve activity data. DeepDTA data preparation aborted.")
        # Provide fallback option
        logger.error("Alternative: If API errors persist, you may manually load the previously generated file 'data/processed/final_combined_dataset.csv' to bypass the API and proceed with DeepDTA model training.")
        return

    # 3. Build DeepDTA encoding dictionary
    encoding_data = build_deepdta_encoding(df, protein_sequence)
    
    # 4. Final data organization
    
    # Add protein sequence to DataFrame
    df['target_sequence'] = protein_sequence
    df['target_chembl_id'] = TARGET_CHEMBL_ID
    df['uniprot_id'] = TARGET_UNIPROT_ID
    
    # Reorder and save final DeepDTA dataset
    df = df[['molecule_chembl_id', 'canonical_smiles', 'target_chembl_id', 'uniprot_id', 'target_sequence', 'pIC50']]
    df.to_csv(FINAL_DATA_PATH, index=False)
    
    logger.info(f"Final DeepDTA dataset saved to: {FINAL_DATA_PATH}")
    logger.info(f"SMILES character count: {len(encoding_data['smiles_dict'])}, Protein character count: {len(encoding_data['protein_dict'])}")
    logger.info("DeepDTA data preparation completed.")


if __name__ == "__main__":
    main()
