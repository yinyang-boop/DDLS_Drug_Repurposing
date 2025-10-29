import os
import json
import pandas as pd
import pubchempy as pcp
from rdkit import Chem
# Importing MolecularAnalyzer, SequenceAnalyzer, DeepDTAProcessor is already done in the previous block

# Define file paths
DATA_DIR = "/content/drive/MyDrive/DDLS_Drug_Repurposing/DDLS_Drug_Repurposing/data/processed"
ENCODING_FILE = f"{DATA_DIR}/deepdta_encoding.json"
DATASET_FILE = f"{DATA_DIR}/deepdta_dataset.csv"

# Utility function to fetch SMILES (copied from original script analysis)
def fetch_smiles(name=None, cid=None):
    """Fetches canonical SMILES for a compound by name or CID using PubChemPy."""
    try:
        if name:
            compounds = pcp.get_compounds(name, 'name')
        elif cid:
            compounds = pcp.Compound.from_cid(cid)
            compounds = [compounds] if compounds else [] # Ensure it's a list
        else:
            return None

        if compounds:
            # Use connectivity_smiles as canonical_smiles is deprecated
            smiles = compounds[0].connectivity_smiles
            # Basic validation with RDKit
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                return smiles
            else:
                # Print the specific RDKit error if validation fails
                print(f"RDKit validation failed for SMILES of {name or cid}")
                return None
        else:
            print(f"Compound not found for {name or cid}")
            return None
    except Exception as e:
        print(f"Error fetching SMILES for {name or cid}: {e}")
        return None

def main():
    report = []

    report.append("## MCP Tools Test Report")

    # --- Test MolecularAnalyzer ---
    report.append("\n--- Testing MolecularAnalyzer ---")

    # Resolve SMILES for Semaglutide and Tirzepatide using the utility function
    # This needs to happen BEFORE they are used in any subsequent tests.
    sema_smiles = fetch_smiles(name="Semaglutide")
    tirz_smiles = fetch_smiles(cid=156588324) # Tirzepatide CID


    if sema_smiles:
        report.append(f"\nSemaglutide SMILES fetched: {sema_smiles[:50]}...")
        # Calculate descriptors for Semaglutide
        try:
            desc_sema = MolecularAnalyzer.calculate_descriptors(sema_smiles)
            report.append(f"Semaglutide descriptors calculated.")
            # Add assertions for descriptors
            assert isinstance(desc_sema, dict), "Assertion Failed: Semaglutide descriptors should be a dictionary"
            assert len(desc_sema) > 0, "Assertion Failed: Semaglutide descriptors dictionary should not be empty"
            assert 'MW (g/mol)' in desc_sema, "Assertion Failed: MW descriptor missing for Semaglutide"
            assert desc_sema.get('Heavy Atoms', 0) > 0, "Assertion Failed: Heavy Atoms descriptor missing or zero for Semaglutide"
            assert isinstance(desc_sema.get('MW (g/mol)'), (int, float)), "Assertion Failed: MW descriptor should be numerical"
            assert isinstance(desc_sema.get('LogP'), (int, float)), "Assertion Failed: LogP descriptor should be numerical"


        except Exception as e:
            report.append(f"Error calculating Semaglutide descriptors: {e}")
            desc_sema = None
    else:
        report.append("\nFailed to fetch Semaglutide SMILES. Skipping descriptor calculation and similarity tests for Semaglutide.")
        desc_sema = None


    if tirz_smiles:
        report.append(f"Tirzepatide SMILES fetched: {tirz_smiles[:50]}...")
        # Calculate descriptors for Tirzepatide
        try:
            desc_tirz = MolecularAnalyzer.calculate_descriptors(tirz_smiles)
            report.append(f"Tirzepatide descriptors calculated.")
            # Add assertions for descriptors
            assert isinstance(desc_tirz, dict), "Assertion Failed: Tirzepatide descriptors should be a dictionary"
            assert len(desc_tirz) > 0, "Assertion Failed: Tirzepatide descriptors dictionary should not be empty"
            assert 'MW (g/mol)' in desc_tirz, "Assertion Failed: MW descriptor missing for Tirzepatide"
            assert desc_tirz.get('Heavy Atoms', 0) > 0, "Assertion Failed: Heavy Atoms descriptor missing or zero for Tirzepatide"
            assert isinstance(desc_tirz.get('MW (g/mol)'), (int, float)), "Assertion Failed: MW descriptor should be numerical"
            assert isinstance(desc_tirz.get('LogP'), (int, float)), "Assertion Failed: LogP descriptor should be numerical"
        except Exception as e:
            report.append(f"Error calculating Tirzepatide descriptors: {e}")
            desc_tirz = None
    else:
         report.append("Failed to fetch Tirzepatide SMILES. Skipping descriptor calculation and similarity tests for Tirzepatide.")
         desc_tirz = None


    # Calculate Tanimoto similarity if both SMILES are available
    if sema_smiles and tirz_smiles:
        try:
            similarity = MolecularAnalyzer.calculate_tanimoto_similarity(sema_smiles, tirz_smiles)
            report.append(f"Tanimoto similarity (Semaglutide vs Tirzepatide): {similarity:.4f}")
            # Add assertions for Tanimoto similarity
            assert isinstance(similarity, float), "Assertion Failed: Tanimoto similarity should be a float"
            assert 0.0 <= similarity <= 1.0, "Assertion Failed: Tanimoto similarity should be between 0 and 1"
            # Optional: assert similarity is above a certain threshold if expected
            # assert similarity > 0.5, "Tanimoto similarity unexpectedly low"

        except Exception as e:
            report.append(f"Error calculating Tanimoto similarity: {e}")
            similarity = None
    else:
        report.append("Skipping Tanimoto similarity calculation due to missing SMILES.")
        similarity = None


    # Test similarity matrix calculation
    smiles_list = [
        "CC(=O)Oc1ccccc1C(=O)O",   # Aspirin
        "CC(C)NCC(O)COc1cccc2ccccc12",  # Propranolol
        "CC(C)Cc1ccc(O)cc1"  # Isopropylphenol
    ]
    try:
        sim_matrix = MolecularAnalyzer.similarity_matrix(smiles_list)
        report.append("\nSimilarity matrix calculated.")
        # Add assertions for similarity matrix
        assert isinstance(sim_matrix, pd.DataFrame), "Assertion Failed: Similarity matrix should be a pandas DataFrame"
        assert sim_matrix.shape == (len(smiles_list), len(smiles_list)), "Assertion Failed: Similarity matrix has incorrect shape"
        # Check diagonal elements are close to 1.0
        for i in range(len(smiles_list)):
             assert abs(sim_matrix.iloc[i, i] - 1.0) < 1e-6, f"Assertion Failed: Diagonal element {i},{i} is not 1.0"
        # Check symmetry (within tolerance for float comparisons)
        # Note: pd.DataFrame.equals checks values and index/column types/order.
        # For float comparisons, isclose or np.allclose is generally preferred,
        # but equals with default tolerance might be sufficient here depending on RDKit precision.
        # A more robust check would be:
        # assert (sim_matrix - sim_matrix.T).abs().max().max() < 1e-9, "Similarity matrix is not symmetric"
        # Using .equals for simplicity as requested in previous analysis.
        assert sim_matrix.equals(sim_matrix.T), "Assertion Failed: Similarity matrix is not symmetric"

    except Exception as e:
        report.append(f"Error calculating similarity matrix: {e}")
        sim_matrix = None


    # --- Test SequenceAnalyzer ---
    report.append("\n--- Testing SequenceAnalyzer ---")
    # Using the GLP-1 backbone for Semaglutide and a different sequence for Tirzepatide testing
    SEQ_SEMAGLUTIDE = "HAEGTFTSDVSSYLEGQAAKEFIAWLVKGRG"
    SEQ_TIRZEPATIDE_TEST = "ABCDEDCBA" # Different sequence for testing non-identity
    SEQ_IDENTICAL = "ABCABC"
    SEQ_SIMILAR = "AXCBYC"
    SEQ_EMPTY = ""


    try:
        seq_identity = SequenceAnalyzer.sequence_identity(SEQ_SEMAGLUTIDE, SEQ_TIRZEPATIDE_TEST)
        seq_norm_sim = SequenceAnalyzer.normalized_similarity(SEQ_SEMAGLUTIDE, SEQ_TIRZEPATIDE_TEST)
        seq_identity_identical = SequenceAnalyzer.sequence_identity(SEQ_IDENTICAL, SEQ_IDENTICAL)
        seq_norm_sim_identical = SequenceAnalyzer.normalized_similarity(SEQ_IDENTICAL, SEQ_IDENTICAL)
        seq_identity_similar = SequenceAnalyzer.sequence_identity(SEQ_IDENTICAL, SEQ_SIMILAR)
        seq_norm_sim_similar = SequenceAnalyzer.normalized_similarity(SEQ_IDENTICAL, SEQ_SIMILAR)
        seq_identity_empty1 = SequenceAnalyzer.sequence_identity(SEQ_EMPTY, SEQ_SIMILAR)
        seq_norm_sim_empty1 = SequenceAnalyzer.normalized_similarity(SEQ_EMPTY, SEQ_SIMILAR)
        seq_identity_empty2 = SequenceAnalyzer.sequence_identity(SEQ_SIMILAR, SEQ_EMPTY)
        seq_norm_sim_empty2 = SequenceAnalyzer.normalized_similarity(SEQ_SIMILAR, SEQ_EMPTY)
        seq_identity_both_empty = SequenceAnalyzer.sequence_identity(SEQ_EMPTY, SEQ_EMPTY)
        seq_norm_sim_both_empty = SequenceAnalyzer.normalized_similarity(SEQ_EMPTY, SEQ_EMPTY)


        report.append(f"Sequence Identity (Semaglutide GLP-1 vs Test Sequence): {seq_identity:.4f}")
        report.append(f"Normalized Similarity (Semaglutide GLP-1 vs Test Sequence): {seq_norm_sim:.4f}")
        report.append(f"Sequence Identity (Identical): {seq_identity_identical:.4f}")
        report.append(f"Normalized Similarity (Identical): {seq_norm_sim_identical:.4f}")
        report.append(f"Sequence Identity (Similar): {seq_identity_similar:.4f}")
        report.append(f"Normalized Similarity (Similar): {seq_norm_sim_similar:.4f}")
        report.append(f"Sequence Identity (Empty vs Similar): {seq_identity_empty1:.4f}")
        report.append(f"Normalized Similarity (Empty vs Similar): {seq_norm_sim_empty1:.4f}")
        report.append(f"Sequence Identity (Similar vs Empty): {seq_identity_empty2:.4f}")
        report.append(f"Normalized Similarity (Similar vs Empty): {seq_norm_sim_empty2:.4f}")
        report.append(f"Sequence Identity (Both Empty): {seq_identity_both_empty:.4f}")
        report.append(f"Normalized Similarity (Both Empty): {seq_norm_sim_both_empty:.4f}")


        # Add assertions for sequence similarity (different sequences)
        assert isinstance(seq_identity, float), "Assertion Failed: Sequence identity should be a float"
        assert 0.0 <= seq_identity <= 1.0, "Assertion Failed: Sequence identity should be between 0 and 1"
        assert isinstance(seq_norm_sim, float), "Assertion Failed: Normalized similarity should be a float"
        assert 0.0 <= seq_norm_sim <= 1.0, "Assertion Failed: Normalized similarity should be between 0 and 1"
        # Since sequences are different, assert they are not 1.0
        assert seq_identity < 1.0, "Assertion Failed: Sequence identity unexpectedly 1.0 for different sequences"
        assert seq_norm_sim < 1.0, "Assertion Failed: Normalized similarity unexpectedly 1.0 for different sequences"

        # Add assertions for sequence similarity (identical sequences)
        assert seq_identity_identical == 1.0, "Assertion Failed: Sequence identity for identical sequences should be 1.0"
        assert seq_norm_sim_identical == 1.0, "Assertion Failed: Normalized similarity for identical sequences should be 1.0"

        # Add assertions for sequence similarity (empty sequences - edge cases)
        assert seq_identity_empty1 == 0.0, "Assertion Failed: Sequence identity with empty string should be 0.0"
        assert seq_norm_sim_empty1 == 0.0, "Assertion Failed: Normalized similarity with empty string should be 0.0"
        assert seq_identity_empty2 == 0.0, "Assertion Failed: Sequence identity with empty string should be 0.0 (reversed)"
        assert seq_norm_sim_empty2 == 0.0, "Assertion Failed: Normalized similarity with empty string should be 0.0 (reversed)"
        assert seq_identity_both_empty == 1.0, "Assertion Failed: Sequence identity for two empty strings should be 1.0" # Conventionally 1.0
        assert seq_norm_sim_both_empty == 1.0, "Assertion Failed: Normalized similarity for two empty strings should be 1.0" # Conventionally 1.0


    except Exception as e:
        report.append(f"Error testing SequenceAnalyzer: {e}")


    # --- Test DeepDTAProcessor ---
    report.append("\n--- Testing DeepDTAProcessor ---")
    deepdta_summary = {}

    # Check if encoding file exists
    deepdta_summary['Encoding file found'] = os.path.exists(ENCODING_FILE)
    report.append(f"Checking for ENCODING_FILE: {ENCODING_FILE}")
    report.append(f"Encoding file found: {deepdta_summary['Encoding file found']}")
    assert deepdta_summary['Encoding file found'] is True, f"Assertion Failed: Encoding file not found at {ENCODING_FILE}"

    # Check if dataset file exists
    deepdta_summary['Dataset file found'] = os.path.exists(DATASET_FILE)
    report.append(f"Checking for DATASET_FILE: {DATASET_FILE}")
    report.append(f"Dataset file found: {deepdta_summary['Dataset file found']}")
    assert deepdta_summary['Dataset file found'] is True, f"Assertion Failed: Dataset file not found at {DATASET_FILE}"

    # Test processor instantiation and encoding (only if encoding file exists and sema_smiles is available)
    proc = None
    if deepdta_summary['Encoding file found']:
        try:
            # Instantiate the processor with the correct data directory
            proc = DeepDTAProcessor(data_dir=DATA_DIR)
            report.append("DeepDTAProcessor instantiated successfully.")

            # Test SMILES encoding with a valid SMILES
            # Use the fetched sema_smiles here
            if sema_smiles:
                try:
                    encoded = proc.encode_smiles(sema_smiles, max_len=50)
                    deepdta_summary['Encoded sample length'] = len(encoded)
                    report.append(f"Encoded Semaglutide length: {deepdta_summary['Encoded sample length']}")
                    # Add assertion for encoded length and type
                    assert isinstance(deepdta_summary['Encoded sample length'], int), "Assertion Failed: Encoded length should be an integer"
                    assert deepdta_summary['Encoded sample length'] == 50, "Assertion Failed: Encoded length is not the expected max_len"
                    assert isinstance(encoded, list), "Assertion Failed: Encoded output should be a list"
                    assert all(isinstance(x, int) for x in encoded), "Assertion Failed: Encoded elements should be integers"
                except Exception as e:
                    report.append(f"Error encoding Semaglutide SMILES: {e}")
                    deepdta_summary['Encoded sample length'] = "Encoding Error"

            else:
                report.append("Skipping SMILES encoding test as Semaglutide SMILES was not fetched.")
                deepdta_summary['Encoded sample length'] = "Skipped"

            # Test encoding with invalid/empty inputs
            try:
                # Test encoding invalid SMILES - Expecting ValueError from RDKit internally
                try:
                    proc.encode_smiles("InvalidSMILES", max_len=50)
                    # If no exception is raised, this is unexpected based on previous runs
                    report.append("Assertion Failed: Encoding invalid SMILES did not raise ValueError as expected.")
                    assert False, "Expected ValueError for invalid SMILES"
                except ValueError as e:
                    # If ValueError is raised, the test passes
                    report.append(f"Encoding InvalidSMILES correctly raised ValueError: {e}")
                    assert True # Test passes if ValueError is raised

                # Test encoding empty SMILES
                try:
                    encoded_empty = proc.encode_smiles("", max_len=50)
                    report.append(f"Encoded Empty SMILES length: {len(encoded_empty)}")
                    # Assuming encoding empty SMILES results in padding tokens
                    assert len(encoded_empty) == 50, "Assertion Failed: Encoded empty SMILES length incorrect"
                    assert all(isinstance(x, int) for x in encoded_empty), "Assertion Failed: Encoded empty elements should be integers"
                    # assert all(x == proc.padding_token_id for x in encoded_empty), "Encoded empty SMILES not all padding" # Requires access to padding token ID

                except ValueError as e:
                    report.append(f"Encoding Empty SMILES incorrectly raised ValueError: {e}")
                    assert False, "Encoding empty SMILES should not raise ValueError"


            except Exception as e:
                 report.append(f"Error testing DeepDTAProcessor encoding with invalid/empty inputs: {e}")


            # Test protein encoding (optional, if needed) - Add assertions if implemented
            # seq = "MTEYKLVVVGAGGVGKSALTIQLIQNHFVDEYDPT"
            # encoded_seq = proc.encode_protein(seq, max_len=50)
            # report.append(f"Encoded protein length: {len(encoded_seq)}")
            # assert isinstance(len(encoded_seq), int), "Encoded protein length should be an integer"
            # assert len(encoded_seq) == 50, "Encoded protein length is not the expected max_len"


        except Exception as e:
            report.append(f"Error testing DeepDTAProcessor instantiation or encoding: {e}")

    else:
        report.append("Skipping DeepDTAProcessor instantiation and encoding tests due to missing encoding file.")


    # Test data cleaning and splitting (only if dataset file exists)
    df = None
    df_clean = None
    train, val, test = None, None, None
    sizes = None # Initialize sizes here

    if deepdta_summary['Dataset file found']:
         # Reading DATASET_FILE: /content/drive/MyDrive/DDLS_Drug_Repurposing/DDLS_Drug_Repurposing/data/processed/deepdta_dataset.csv
         try:
             df = pd.read_csv(DATASET_FILE)
             report.append(f"Dataset loaded. Original shape: {df.shape}")
             assert isinstance(df, pd.DataFrame), "Assertion Failed: Loaded dataset should be a pandas DataFrame"
             assert not df.empty, "Assertion Failed: Loaded dataset should not be empty"
             assert all(col in df.columns for col in ['canonical_smiles', 'target_sequence', 'pIC50']), "Assertion Failed: Dataset missing required columns"

             df_clean = DeepDTAProcessor.clean_dta_dataframe(df.copy()) # Use a copy to avoid modifying original df if needed later
             report.append(f"Dataset cleaned. Cleaned shape: {df_clean.shape}")
             assert isinstance(df_clean, pd.DataFrame), "Assertion Failed: Cleaned dataset should be a pandas DataFrame"
             # Assertions about cleaning effects (e.g., no NaNs in key columns, valid pIC50 range)
             assert df_clean[['canonical_smiles', 'target_sequence', 'pIC50']].isnull().sum().sum() == 0, "Assertion Failed: Cleaned data contains NaNs in key columns"
             assert (df_clean['pIC50'] >= 0).all(), "Assertion Failed: Cleaned pIC50 values are not all non-negative"
             assert df_clean.shape[0] <= df.shape[0], "Assertion Failed: Cleaned dataframe has more rows than original" # Assumes cleaning might drop rows


             train, val, test = DeepDTAProcessor.split_dataset(df_clean)
             report.append(f"Dataset split. Train/Val/Test shapes: {train.shape}/{val.shape}/{test.shape}")
             # Add assertions for splits
             assert isinstance(train, pd.DataFrame) and isinstance(val, pd.DataFrame) and isinstance(test, pd.DataFrame), "Assertion Failed: Split outputs should be DataFrames"
             assert train.shape[0] + val.shape[0] + test.shape[0] == df_clean.shape[0], "Assertion Failed: Sum of split rows does not equal cleaned dataframe rows"
             # Assert approximate split ratios (e.g., 80/10/10) - calculate expected rows based on df_clean size
             total_rows = df_clean.shape[0]
             expected_train_rows = int(total_rows * 0.8)
             expected_val_rows = int(total_rows * 0.1)
             expected_test_rows = total_rows - expected_train_rows - expected_val_rows # The remainder goes to test

             # Allow for small variations in split sizes due to rounding or stratification
             assert abs(train.shape[0] - expected_train_rows) <= 2, f"Assertion Failed: Train split size mismatch. Expected approx {expected_train_rows}, got {train.shape[0]}"
             assert abs(val.shape[0] - expected_val_rows) <= 2, f"Assertion Failed: Validation split size mismatch. Expected approx {expected_val_rows}, got {val.shape[0]}"
             assert abs(test.shape[0] - expected_test_rows) <= 2, f"Assertion Failed: Test split size mismatch. Expected approx {expected_test_rows}, got {test.shape[0]}"
             assert train.shape[1] == val.shape[1] == test.shape[1] == df_clean.shape[1], "Assertion Failed: Split dataframes have different number of columns"

             # Store split sizes in summary for report
             deepdta_summary["Train/Val/Test"] = f"{train.shape[0]}/{val.shape[0]}/{test.shape[0]}"
             # The following lines were causing IndentationError, let's ensure they are correctly indented
             sizes = {"Train": train.shape[0], "Val": val.shape[0], "Test": test.shape[0]}
             # report.append(f"Split sizes: {sizes}") # Removed redundant print


             # Test cleaning with invalid/edge case dataframes
             # Create a dataframe with missing values and invalid pIC50
             invalid_data = {'molecule_chembl_id': ['CHEMBL1', 'CHEMBL2', 'CHEMBL3', 'CHEMBL4', 'CHEMBL5'],
                             'canonical_smiles': ['CCO', None, 'CCN', 'C1=CC=CC=C1', 'InvalidSMILES'], # Test None and invalid SMILES
                             'target_chembl_id': ['TARGET1', 'TARGET2', 'TARGET3', 'TARGET4', 'TARGET5'],
                             'uniprot_id': ['UNIPROT1', 'UNIPROT2', 'UNIPROT3', 'UNIPROT4', 'UNIPROT5'],
                             'target_sequence': ['SEQ1', 'SEQ2', None, 'SEQ4', 'SEQ5'], # Test None sequence
                             'pIC50': [7.0, -5.0, None, 8.0, 9.0]} # Test negative and None pIC50
             invalid_df = pd.DataFrame(invalid_data)

             try:
                  invalid_df_clean = DeepDTAProcessor.clean_dta_dataframe(invalid_df.copy())
                  report.append(f"\nTested cleaning with invalid data. Original shape: {invalid_df.shape}, Cleaned shape: {invalid_df_clean.shape}")
                  assert isinstance(invalid_df_clean, pd.DataFrame), "Assertion Failed: Cleaning invalid data should return DataFrame"
                  # Assert that rows with None/invalid pIC50 and None smiles/sequence are removed
                  # Assuming cleaning removes rows with None in canonical_smiles, target_sequence, or pIC50 < 0 or None pIC50, AND potentially invalid SMILES if RDKit validation is part of cleaning.
                  # Based on typical cleaning, CHEMBL2, CHEMBL3 should be removed. If invalid SMILES are also filtered, CHEMBL5 should be removed.
                  # Let's assume only rows with None in key columns or negative pIC50 are removed by clean_dta_dataframe based on previous task analysis.
                  # Expected: CHEMBL1, CHEMBL4, CHEMBL5 remain. Shape should be (3, 6).
                  assert invalid_df_clean.shape[0] == 3, f"Assertion Failed: Cleaning invalid data resulted in incorrect number of rows ({invalid_df_clean.shape[0]})"
                  assert all(id in invalid_df_clean['molecule_chembl_id'].tolist() for id in ['CHEMBL1', 'CHEMBL4', 'CHEMBL5']), "Assertion Failed: Incorrect rows retained after cleaning invalid data"
                  assert invalid_df_clean[['canonical_smiles', 'target_sequence', 'pIC50']].isnull().sum().sum() == 0, "Assertion Failed: Cleaned invalid data contains NaNs"
                  assert (invalid_df_clean['pIC50'] >= 0).all(), "Assertion Failed: Cleaned invalid data contains negative pIC50"

             except Exception as e:
                  report.append(f"Error testing DeepDTAProcessor cleaning with invalid data: {e}")


         except Exception as e:
             report.append(f"Error testing DeepDTAProcessor cleaning or splitting: {e}")
             # If an error occurs here, set sizes to None or handle appropriately
             deepdta_summary["Train/Val/Test"] = "Error during split"
             sizes = None


    else:
        report.append("Skipping DeepDTAProcessor cleaning and splitting tests due to missing dataset file.")
        deepdta_summary["Train/Val/Test"] = "Dataset file not found"
        sizes = None


    # --- Generate Report ---
    report.append("\n--- Test Summary ---")
    # Add summary data to report
    if 'desc_sema' in locals() and desc_sema:
         report.append("\nSemaglutide Descriptors:")
         for key, value in desc_sema.items():
              report.append(f"  - {key}: {value}")
    if 'desc_tirz' in locals() and desc_tirz:
         report.append("\nTirzepatide Descriptors:")
         for key, value in desc_tirz.items():
              report.append(f"  - {key}: {value}")
    if 'similarity' in locals() and similarity is not None:
         report.append(f"\nTanimoto Similarity (Semaglutide vs Tirzepatide): {similarity:.4f}")

    if 'sim_matrix' in locals() and sim_matrix is not None:
         report.append("\nSimilarity Matrix:")
         report.append(sim_matrix.to_string()) # Use to_string for better formatting in report

    if 'seq_identity' in locals() and 'seq_norm_sim' in locals():
         report.append("\nSequence Similarity:")
         report.append(f"Sequence Identity (Semaglutide GLP-1 vs Test Sequence): {seq_identity:.4f}")
         report.append(f"Normalized Similarity (Semaglutide GLP-1 vs Test Sequence): {seq_norm_sim:.4f}")
         report.append(f"Sequence Identity (Identical): {seq_identity_identical:.4f}")
         report.append(f"Normalized Similarity (Identical): {seq_norm_sim_identical:.4f}")
         report.append(f"Sequence Identity (Similar): {seq_identity_similar:.4f}")
         report.append(f"Normalized Similarity (Similar): {seq_norm_sim_similar:.4f}")
         report.append(f"Sequence Identity (Empty vs Similar): {seq_identity_empty1:.4f}")
         report.append(f"Normalized Similarity (Empty vs Similar): {seq_norm_sim_empty1:.4f}")
         report.append(f"Sequence Identity (Similar vs Empty): {seq_identity_empty2:.4f}")
         report.append(f"Normalized Similarity (Similar vs Empty): {seq_norm_sim_empty2:.4f}")
         report.append(f"Sequence Identity (Both Empty): {seq_identity_both_empty:.4f}")
         report.append(f"Normalized Similarity (Both Empty): {seq_norm_sim_both_empty:.4f}")


    report.append("\nDeepDTAProcessor Summary:")
    for key, value in deepdta_summary.items():
         report.append(f"  - {key}: {value}")
    if sizes: # Check if sizes was successfully populated
        report.append(f"  - Split sizes: Train={sizes['Train']}, Val={sizes['Val']}, Test={sizes['Test']}")


    # Write report to file
    report_file_path = "TEST_REPORT_WITH_ASSERTIONS.md" # Use a new name to distinguish
    try:
        with open(report_file_path, "w") as f:
            f.write("\n".join(report))
        print(f"\nTest report written to {report_file_path}")
    except Exception as e:
        print(f"\nError writing test report: {e}")


if __name__ == "__main__":
    main()