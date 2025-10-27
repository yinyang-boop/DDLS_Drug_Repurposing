**DDLS 2025 Final Project: Reproducible Drug-Target Affinity (DTA) Prediction Pipeline**

**Project Overview**
This project develops a Modular Computational Pipeline (MCP) for predicting Drug-Target Affinity (pIC50 values) using a deep learning model. We utilize the DeepDTA (1D CNN) architecture for high-accuracy predictions, focusing heavily on engineering robustness using Prefect 2.x for workflow orchestration and RDKit for standardized cheminformatics.
The project is built around the principles of FAIR Data and Open Science, ensuring high reproducibility and accessibility for SciLifeLab researchers and the broader community.

Topic: ddls-course-2025🚀 

**Key Deliverables and Accessibility**
1. Trained Model Weights (Accessibility) The final trained DeepDTA model, achieving a Concordance Index (Ci) of ~0.85 on the test set, is available for direct download. 
2. The Minimal MCP Toolset (mcp-pipeline) The core chemical analysis and data processing utilities are packaged into the installable mcp-pipeline. This package allows any user to standardize SMILES, calculate descriptors, and encode data for DTA models. Core Tool: mcp_pipeline.MolecularAnalyzer (RDKit wrappers). Core Tool: mcp_pipeline.DeepDTAProcessor (Encoding/Cleaning utilities)
5. Full Project Report (final_project_report.md)
6. AI Deep Research Log (ai_debugging_prompts.md)

**Setup and Installation**
1. Prerequisites
Python 3.8+The rdkit-pypi package must be installed globally or in your environment for the core tools to work.
2. Local Installation
Clone the repository and install the core dependencies, including Prefect and PyTorch.

# Clone the repository
git clone [https://github.com/yinyang-boop/DDLS_Drug_Repurposing.git](https://github.com/yinyang-boop/DDLS_Drug_Repurposing.git)
cd DDLS_Drug_Repurposing

# Install core dependencies (Prefect, PyTorch, Pandas, etc.)
pip install -r requirements.txt

# Install the MCP toolset in development mode
# This makes the mcp_pipeline module available for import
pip install -e .

**Workflow Execution and Inference**
1. Running the Prefect Flow (Full Pipeline)The entire project is orchestrated by a single Prefect Flow defined in src/main_pipeline_run.py. To execute the complete, resilient, and monitored workflow:# Ensure Prefect is running (optional, but recommended for UI monitoring)
# prefect server start 

# Run the main pipeline flow
python src/main_pipeline_run.py

# Monitor the flow's progress, logs, and state history via the Prefect UI.
Output: This will generate the trained model in the /models directory and log performance metrics. 

2. Running Inference with the mcp-pipeline (Minimal Toolset Usage)You can use the modular toolset directly in any script or notebook for quick analysis or inference on new data.Example: Analyzing a New Drug-Target PairFirst, ensure you have the required encoding dictionary and model weights downloaded.# python src/run_inference.py 
# (Assuming you create this script for a dedicated inference endpoint)

import torch
import pandas as pd
from mcp_pipeline import MolecularAnalyzer, DeepDTAProcessor 
from src.model_architecture import DeepDTA # Assuming model class is here

# --- SETUP ---
# 1. Load the trained model (Ensure deepdta_model_final.pt is in /models)
model = DeepDTA() 
model.load_state_dict(torch.load('models/deepdta_model_final.pt'))
model.eval()

# 2. Initialize the Processor
dta_processor = DeepDTAProcessor(encoding_file_path='deepdta_encoding.json')

# --- INFERENCE ---
# Drug: Aspirin
NEW_DRUG_SMILES = "CC(=O)Oc1ccccc1C(=O)O"
# Target: COX-1 (P23230) - Example sequence
NEW_TARGET_SEQ = "MRPSGTAGAALLALLAALCPASRALEEKKVCQGTSNKLTQLGTFEDHFLSLQRMFNNCEVVLGNLEITYVQRNYDL..." 

# A. Encode the inputs
drug_encoded = dta_processor.encode_smiles(NEW_DRUG_SMILES, max_len=100)
target_encoded = dta_processor.encode_protein(NEW_TARGET_SEQ, max_len=1000)

# B. Run Prediction
# NOTE: Convert encoded lists to PyTorch tensors and run through the model
# predicted_pIC50 = model(drug_tensor, target_tensor)

print("\n--- Cheminformatics Analysis (using MolecularAnalyzer) ---")
descriptors = MolecularAnalyzer.calculate_descriptors(NEW_DRUG_SMILES)
print(f"Descriptors for Aspirin: {descriptors}")
# print(f"Predicted pIC50: {predicted_pIC50.item():.2f}")

🤝 Contribution and License
We welcome contributions and feedback, especially for implementing the GNN/Transformer architectures mentioned in the future directions.
License: This project is released under the MIT License.
Issue Reporting: Please use the provided [Bug Report] and [Feature Request] templates in the .github/ISSUE_TEMPLATE directory to maintain clarity and structure.
