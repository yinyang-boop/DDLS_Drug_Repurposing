#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Full Data Pipeline Script – Integration of ChEMBL, PubChem, and BindingDB
Automated data acquisition, cleaning, and standardization for drug repurposing research
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
from rdkit.Chem import PandasTools
import requests
import json
import time
import logging
import warnings
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from tqdm.auto import tqdm
from chembl_webresource_client.new_client import new_client
import pubchempy as pcp

# Configure logging and directory paths
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

# Create folder structure
Path("data/raw").mkdir(parents=True, exist_ok=True)
Path("data/processed").mkdir(parents=True, exist_ok=True)
Path("reports/figures").mkdir(parents=True, exist_ok=True)

class UnifiedDataPipeline:
    """
    Unified data pipeline integrating ChEMBL, PubChem, and BindingDB.
    Provides complete functionality for data acquisition, cleaning, standardization, and visualization.
    """
    
    def __init__(self, cache_dir: str = "./cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Initialize API clients
        self.chembl_client = new_client
        self.session = requests.Session()
        
        # Data storage
        self.df_chembl = None
        self.df_pubchem = None
        self.df_bindingdb = None
        self.df_combined = None
        
        # Plot style
        plt.style.use('seaborn-v0_8-whitegrid')
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    def fetch_chembl_data(self, target_chembl_id: str = "CHEMBL203", 
                          assay_type: str = "B", 
                          standard_type: str = "IC50") -> pd.DataFrame:
        """
        Retrieve bioactivity data for a specific target from the ChEMBL database.
        
        Args:
            target_chembl_id: ChEMBL Target ID (default: EGFR - CHEMBL203)
            assay_type: Assay type (B = binding assay)
            standard_type: Measurement type (IC50, Ki, Kd, etc.)
            
        Returns:
            DataFrame containing ChEMBL bioactivity data
        """
        logger.info(f"Fetching data for target {target_chembl_id} from ChEMBL")
        
        try:
            # Get target information
            target = self.chembl_client.target.filter(target_chembl_id=target_chembl_id).only(
                'target_chembl_id', 'pref_name', 'organism', 'target_type'
            )[0]
            logger.info(f"Target info: {target['pref_name']} ({target['organism']})")
            
            # Get bioactivity data
            activities = self.chembl_client.activity.filter(
                target_chembl_id=target_chembl_id,
                assay_type=assay_type,
                standard_type=standard_type,
                relation="="  # Only retrieve exact measurements
            ).only(
                'activity_id', 'assay_chembl_id', 'molecule_chembl_id',
                'standard_type', 'standard_value', 'standard_units',
                'relation', 'target_chembl_id', 'pchembl_value'
            )
            
            activities_list = list(activities)
            df = pd.DataFrame(activities_list)
            
            if df.empty:
                logger.warning(f"No activity data found for target {target_chembl_id}")
                return pd.DataFrame()
            
            logger.info(f"Retrieved {len(df)} activity records from ChEMBL")
            
            # Get compound structure information
            molecule_ids = df['molecule_chembl_id'].unique().tolist()
            compounds_data = []
            
            for mol_id in tqdm(molecule_ids, desc="Fetching compound structures"):
                try:
                    compound = self.chembl_client.molecule.get(mol_id)
                    compounds_data.append({
                        'molecule_chembl_id': mol_id,
                        'compound_name': compound.get('pref_name', ''),
                        'canonical_smiles': compound.get('molecule_structures', {}).get('canonical_smiles', ''),
                        'molecular_weight': compound.get('molecular_weight', None),
                        'alogp': compound.get('alogp', None)
                    })
                    time.sleep(0.1)
                except Exception as e:
                    logger.warning(f"Failed to get info for compound {mol_id}: {e}")
                    continue
            
            compounds_df = pd.DataFrame(compounds_data)
            
            # Merge activity and compound data
            if not compounds_df.empty:
                df = pd.merge(df, compounds_df, on='molecule_chembl_id', how='left')
            
            df.to_csv(f"data/raw/chembl_{target_chembl_id}_raw.csv", index=False)
            self.df_chembl = df
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to fetch ChEMBL data: {e}")
            return pd.DataFrame()
    
    def fetch_pubchem_data(self, compound_list: List[str], 
                          properties: List[str] = None) -> pd.DataFrame:
        """
        Retrieve compound data from PubChem.
        
        Args:
            compound_list: List of compound names or CIDs
            properties: List of properties to retrieve
            
        Returns:
            DataFrame containing PubChem compound data
        """
        if properties is None:
            properties = ['cid', 'iupac_name', 'canonical_smiles', 'isomeric_smiles',
                         'molecular_weight', 'molecular_formula', 'xlogp',
                         'hydrogen_bond_donor_count', 'hydrogen_bond_acceptor_count',
                         'rotatable_bond_count', 'complexity']
        
        logger.info(f"Fetching PubChem data for {len(compound_list)} compounds")
        
        results = []
        failed_compounds = []
        
        for compound_id in tqdm(compound_list, desc="Querying PubChem"):
            try:
                if isinstance(compound_id, str) and compound_id.isdigit():
                    compound = pcp.Compound.from_cid(int(compound_id))
                else:
                    compounds = pcp.get_compounds(compound_id, 'name')
                    if not compounds:
                        compounds = pcp.get_compounds(compound_id, 'smiles')
                    if compounds:
                        compound = compounds[0]
                    else:
                        failed_compounds.append(compound_id)
                        continue
                
                compound_data = {'query_id': compound_id}
                for prop in properties:
                    try:
                        compound_data[prop] = getattr(compound, prop, None)
                    except AttributeError:
                        compound_data[prop] = None
                
                results.append(compound_data)
                time.sleep(0.2)
                
            except Exception as e:
                logger.warning(f"Failed to retrieve info for compound {compound_id}: {e}")
                failed_compounds.append(compound_id)
                continue
        
        if failed_compounds:
            logger.warning(f"Failed to retrieve info for {len(failed_compounds)} compounds")
        
        df = pd.DataFrame(results)
        
        if not df.empty:
            df.to_csv("data/raw/pubchem_data_raw.csv", index=False)
            self.df_pubchem = df
        
        return df
    
    def fetch_bindingdb_data(self, target_uniprot_id: str = "P00533") -> pd.DataFrame:
        """
        Retrieve binding affinity data from BindingDB for a specific UniProt target.
        
        Args:
            target_uniprot_id: UniProt ID (default: EGFR - P00533)
            
        Returns:
            DataFrame containing BindingDB binding data
        """
        logger.info(f"Fetching BindingDB data for UniProt ID {target_uniprot_id}")
        
        try:
            # Method 1: Using BindingDB API
            url = f"http://www.bindingdb.org/axis2/services/BDBService/getLigandsByUniprots"
            params = {
                'uniprot': target_uniprot_id,
                'code': 0,
                'response': 'application/json'
            }
            
            response = self.session.get(url, params=params, timeout=30)
            
            if response.status_code == 200:
                try:
                    data = response.json()
                    if data and 'getLigandsByUniprots' in data:
                        ligands = data['getLigandsByUniprots']
                        df = pd.DataFrame(ligands)
                        logger.info(f"Retrieved {len(df)} records via API")
                        return df
                except json.JSONDecodeError:
                    logger.warning("API returned non-JSON format, trying backup method")
            
            # Method 2: Backup web-scraping
            return self._fetch_bindingdb_backup(target_uniprot_id)
            
        except Exception as e:
            logger.error(f"Failed to fetch BindingDB data: {e}")
            return self._fetch_bindingdb_backup(target_uniprot_id)
    
    def _fetch_bindingdb_backup(self, target_uniprot_id: str) -> pd.DataFrame:
        """Backup BindingDB data retrieval method"""
        try:
            from bs4 import BeautifulSoup
            
            url = f"https://www.bindingdb.org/rwd/bind/chemsearch/marvin/SKetch.jsp"
            params = {
                'uniprotAcc': target_uniprot_id,
                'max_rows': 1000
            }
            
            response = self.session.get(url, params=params, timeout=60)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            logger.warning("Backup BindingDB parsing logic needs further implementation")
            return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"Backup BindingDB method also failed: {e}")
            return pd.DataFrame()
