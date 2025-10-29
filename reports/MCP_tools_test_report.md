## MCP Tools Test Report

--- Testing MolecularAnalyzer ---

Semaglutide SMILES fetched: CCC(C)C(C(=O)NC(C)C(=O)NC(CC1=CNC2=CC=CC=C21)C(=O)...
Semaglutide descriptors calculated.
Tirzepatide SMILES fetched: CCC(C)C(C(=O)NC(C)C(=O)NC(CCC(=O)N)C(=O)NC(CCCCN)C...
Tirzepatide descriptors calculated.
Tanimoto similarity (Semaglutide vs Tirzepatide): 0.7150

Similarity matrix calculated.

--- Testing SequenceAnalyzer ---
Sequence Identity (Semaglutide GLP-1 vs Test Sequence): 0.0000
Normalized Similarity (Semaglutide GLP-1 vs Test Sequence): 0.1290
Sequence Identity (Identical): 1.0000
Normalized Similarity (Identical): 1.0000
Sequence Identity (Similar): 0.5000
Normalized Similarity (Similar): 0.5000
Sequence Identity (Empty vs Similar): 0.0000
Normalized Similarity (Empty vs Similar): 0.0000
Sequence Identity (Similar vs Empty): 0.0000
Normalized Similarity (Similar vs Empty): 0.0000
Sequence Identity (Both Empty): 0.0000
Normalized Similarity (Both Empty): 0.0000
Error testing SequenceAnalyzer: Assertion Failed: Sequence identity for two empty strings should be 1.0

--- Testing DeepDTAProcessor ---
Checking for ENCODING_FILE: /content/drive/MyDrive/DDLS_Drug_Repurposing/DDLS_Drug_Repurposing/data/processed/deepdta_encoding.json
Encoding file found: True
Checking for DATASET_FILE: /content/drive/MyDrive/DDLS_Drug_Repurposing/DDLS_Drug_Repurposing/data/processed/deepdta_dataset.csv
Dataset file found: True
Error testing DeepDTAProcessor instantiation or encoding: DeepDTAProcessor.__init__() got an unexpected keyword argument 'data_dir'
Dataset loaded. Original shape: (13286, 6)
Dataset cleaned. Cleaned shape: (13286, 6)
Dataset split. Train/Val/Test shapes: (10628, 6)/(1329, 6)/(1329, 6)

Tested cleaning with invalid data. Original shape: (5, 6), Cleaned shape: (3, 6)

--- Test Summary ---

Semaglutide Descriptors:
  - MW (g/mol): 4113.640999999981
  - LogP: -11.627860000000188
  - H Donors: 57
  - H Acceptors: 56
  - TPSA: 1646.1799999999994
  - Rotatable Bonds: 149
  - Heavy Atoms: 291

Tirzepatide Descriptors:
  - MW (g/mol): 4813.52699999999
  - LogP: -12.09750000000029
  - H Donors: 58
  - H Acceptors: 65
  - TPSA: 1789.629999999999
  - Rotatable Bonds: 163
  - Heavy Atoms: 341

Tanimoto Similarity (Semaglutide vs Tirzepatide): 0.7150

Similarity Matrix:
                             CC(=O)Oc1ccccc1C(=O)O  CC(C)NCC(O)COc1cccc2ccccc12  CC(C)Cc1ccc(O)cc1
CC(=O)Oc1ccccc1C(=O)O                     1.000000                     0.204082           0.131579
CC(C)NCC(O)COc1cccc2ccccc12               0.204082                     1.000000           0.173913
CC(C)Cc1ccc(O)cc1                         0.131579                     0.173913           1.000000

Sequence Similarity:
Sequence Identity (Semaglutide GLP-1 vs Test Sequence): 0.0000
Normalized Similarity (Semaglutide GLP-1 vs Test Sequence): 0.1290
Sequence Identity (Identical): 1.0000
Normalized Similarity (Identical): 1.0000
Sequence Identity (Similar): 0.5000
Normalized Similarity (Similar): 0.5000
Sequence Identity (Empty vs Similar): 0.0000
Normalized Similarity (Empty vs Similar): 0.0000
Sequence Identity (Similar vs Empty): 0.0000
Normalized Similarity (Similar vs Empty): 0.0000
Sequence Identity (Both Empty): 0.0000
Normalized Similarity (Both Empty): 0.0000

DeepDTAProcessor Summary:
  - Encoding file found: True
  - Dataset file found: True
  - Train/Val/Test: 10628/1329/1329
  - Split sizes: Train=10628, Val=1329, Test=1329