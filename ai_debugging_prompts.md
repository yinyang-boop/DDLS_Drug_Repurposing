# AI-Assisted Debugging Prompts for Drug Repurposing Project

## Phase 1: Data Acquisition Debugging

### Prompt 1.1 (Initial Error Detection)
I am running a Python script to fetch data from the [ChEMBL/PubChem/BindingDB] API but I am encountering an error. The error message is: [PASTE ERROR LOG]. The code snippet around the error is: [PASTE RELEVANT CODE]. Please analyze the error and suggest the most common fixes.

### Prompt 1.2 (If API Request Fails)
The API request to [DATABASE NAME] is failing with status code [STATUS CODE, e.g., 403/504]. I have already checked [BASIC STEPS, e.g., internet connection, API quota]. Based on the error pattern, suggest:
1. Immediate steps to mitigate (e.g., implementing a retry mechanism)
2. Alternative endpoints or data sources
3. A code example to handle this specific exception gracefully

### Prompt 1.3 (If Data Format is Invalid)
The data fetched from [DATABASE NAME] does not match the expected schema. Expected fields: [LIST EXPECTED FIELDS, e.g., "smiles", "affinity"]. Actual response snippet: [PASTE ACTUAL RESPONSE SNIPPET]. Please generate a data validation function that:
1. Checks for mandatory fields
2. Handles type conversions (e.g., string to float)
3. Logs mismatches for further inspection

## Phase 2: Data Cleaning & Standardization Debugging (RDKit & DeepDTA Encoding)

### Prompt 2.1 (When RDKit/Chem Informatics Fails)
I encountered an error like `AttributeError: module 'rdkit.Chem' has no attribute '...'` or other SMILES processing failure when using the RDKit function [SPECIFIC FUNCTION, e.g., GetMorganFingerprint]. Input SMILES: [EXAMPLE SMILES]. Error: [ERROR MESSAGE]. Please suggest:
1. RDKit version compatibility issues (e.g., importing from `rdkit.Chem` vs `rdkit.Chem.AllChem`)
2. Common SMILES format issues and corresponding standardization methods (e.g., deprotonation/normalization)
3. Code examples for salt removal using `rdkit.Chem.SaltRemover`

### Prompt 2.2 (DeepDTA Encoding Fails)
I failed to encode SMILES or protein sequences into the numerical vectors required by the DeepDTA model. SMILES/Sequence example: [INPUT EXAMPLE]. Expected length: [VALUE]. Actual output: [OUTPUT SNIPPET]. Please diagnose:
1. If the encoding dictionaries (`smiles_chars`/`protein_chars`) are missing characters
2. If the truncation or padding logic is correctly applied to the fixed input length
3. Whether the encoding function should use One-Hot Encoding or a Lookup Table approach

### Prompt 2.3 (When Dataset is Too Small)
My cleaned dataset only has [NUMBER] samples, which is insufficient for DeepDTA training. Suggest data augmentation techniques such as:
- Augmenting the molecular graph for drugs (e.g., atom masking)
- Using transfer learning from larger datasets (e.g., pre-training on ZINC)
- Generating synthetic DTA data using cGANs (Conditional Generative Adversarial Networks)

## Phase 3: Model Training Debugging

### Prompt 3.1 (When Loss Does Not Converge)
My [MODEL TYPE, DeepDTA CNN] is training on DTA data, but the loss is not converging. Here are the last 10 loss values: [LIST]. The current hyperparameters are: [LEARNING RATE, BATCH SIZE, etc.]. Please suggest:
1. Hyperparameter tuning ranges
2. Alternative optimizer settings (e.g., AdamW)
3. Gradient clipping or learning rate scheduling schemes
4. Checks on the model structure, especially 1D CNN kernel size and dense layer design

### Prompt 3.2 (When GPU Memory is Insufficient)
I am getting CUDA out-of-memory errors when training on [HARDWARE CONFIGURATION]. The model architecture is: [BRIEF MODEL STRUCTURE]. Current batch size: [VALUE]. Propose:
1. Memory-efficient alternatives (e.g., gradient accumulation)
2. Model quantification or Mixed-Precision training configurations (`torch.cuda.amp`)
3. Optimization of DataLoader settings (e.g., `num_workers` and `pin_memory`)

## Phase 4: Model Evaluation & Prediction Debugging

### Prompt 4.1 (When Test Results Are Anomalous)
Model predictions on the test set show [PROBLEM DESCRIPTION, e.g., all predictions are close to a constant value]. Training metrics were: [TRAINING METRICS]. Test metrics are: [TEST METRICS]. Diagnose potential causes and suggest:
1. Data leakage checks
2. Reviewing the normalization/denormalization process for the target variable (pIC50)
3. Alternative evaluation metrics (e.g., Concordance Index Ci)

### Prompt 4.2 (When Real-World Predictions Are Poor)
The model performs well on test data but fails in real-world drug repurposing scenarios. Example: Predicting affinity for drug [NAME] and target [NAME] gives [PREDICTED VALUE], but the expected value is [ACTUAL VALUE]. Request:
1. Methods to identify Domain Shift
2. Techniques for model calibration
3. Strategies to incorporate uncertainty estimation into prediction results

## **Phase 5: Workflow/Orchestration Debugging (Prefect)**

### Prompt 5.1 (Prefect Flow Registration/Execution Fails)
My Prefect Flow fails to register or execute. Error log: [PASTE ERROR LOG]. The Flow definition snippet from my `main_pipeline.py` is: [PASTE RELEVANT CODE]. Please check:
1. If the Prefect deployment configuration is correct (e.g., using `deployment.yaml`)
2. If the Flow entry function and Task definitions comply with the Prefect 2.x API
3. If containerization or environment dependencies (e.g., RDKit dependencies in the Docker image) are missing

### Prompt 5.2 (Pipeline Task Failure & State Management)
My Prefect Flow failed at the [SPECIFIC TASK NAME] with the error [ERROR MESSAGE]. I want the Flow to automatically retry 3 times upon failure. Please provide:
1. A code example for adding the retry logic to this specific Task
2. Instructions on how to view the Task's logs and input parameters in the Prefect UI
3. How to configure custom failure handling logic (e.g., sending notifications)

## Usage Guidelines
- **Sequential Use**: Start from Phase 1 and proceed sequentially; each Phase's Prompts build upon the solutions of the previous ones.
- **Provide Context**: Always paste relevant error logs, code snippets, and dataset statistics in each interaction.
- **Iterative Refinement**: Adjust the prompt based on the AI agent's response to progressively narrow down the problem.