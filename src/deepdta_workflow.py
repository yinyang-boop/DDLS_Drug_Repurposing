import pandas as pd
import numpy as np
import json
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, MaxPool1D, Flatten, Dense, Concatenate, Embedding
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from typing import List, Dict, Any

# --- 1. Configuration and constants ---

# Assumed maximum sequence lengths for DeepDTA
MAX_SMILES_LEN = 100
MAX_PROTEIN_LEN = 1000
EMBEDDING_DIM = 128

# ❗ Corrected absolute file paths (using user-provided path)
BASE_PATH = '/content/drive/MyDrive/DDLS_Drug_Repurposing/DDLS_Drug_Repurposing/data/processed/'
DATASET_FILE = BASE_PATH + 'deepdta_dataset.csv'
ENCODING_FILE = BASE_PATH + 'deepdta_encoding.json'

# CFTR (P13569) target sequence
CFTR_SEQUENCE = (
    "MQRSPLEKASVVSKLFFSWTRPILRKGYRQRLELSDIYQIPSVDSADNLSEKLEREWDRELSPEYDNQSLESPEYYETLQDICFQKKTCPVQLWNTLAQYSMVIFGEPLQTLQDQMTLEARQTADVGEALETEEWRKVLADQQTRMILCEAWIAQAEAFQDNFESKTTELMIKNFKTRIPTLFKNKFSTLSLGIDLPQSGTFSDKLLTLRDLHHIIIDRNTFRLYTEtLKFIIFGtKGAFGYVCRfSLFESLGEGHRtTEWVfPGQLFPISEGKTTKTTAPVTQENGLWLNHMKYEPQYLFGVGIIVFAGLIVWFTALFQYIVLFGVSGLIVAYFLRKLTLTAFTYVGDLADIEPTESGKISTSIAIAGLIAWLADLAAQFVALLLEGDKLCTLNFrLLEGTFVLIVLMSVLGSNMAVSGVRFSLINPKYAAySLIDDTSLGGLILvSVvSFyFPPVLWLGIIwAILGvALvaGFggLGGLePLVaAFVLyVVlFLILLAgSLSAiLGGLlPVSLGiHTGmLvTLavLGGEGAALSVGLGLTLLaAGlLVlTLIGliKVSYAYQFGLNLTVLQTLHDENIDLSPEmsLQEFNDdEAYLleDkEKEQdKepaYEEyQNLtYlreDkEKEQRKKVlDSleKEtAQyDqntLInESLADsPEYEtLQqIcFQkktCPvqLWnTLAQySMVIfGEPLQTLQDQMTlearQTADvGeALETEEWRkVlADqQTRMILCEAWIAqAeAFqdNfESkTTELmIKnfKtRIPTLFKNFSTlsLGIDLpQSGTFSdKLLtLRdLhHIIIdRNtfRLYtETLKfIIFGtKGAfGYVCRfSLfEsLGEGHRtTEWVfPGQLfPISegkTTkTTapVTqENGLWLNHmKYEpQYLFGvGIIVFAGLIVWFTALFQyIVlFGvSGLIVAYFLRkltLTAFTYVGDLADIEPTESGKISTSIAIgLIAWlAdlAAQfVALlLegaDKlCtlNFRLleGTfVLIvlMSVLGSNMAVSgVRfSLINPkyAAySLIDDTSLGGLILVSVVsfyFppVLWLgIIwAILgvALvAgfggLGGLgeplVAAFvlyVVlFLILLAgSLSAILGGLpVsLGiHTGmLvTLAvLGGEGaAlsVGLGLTlLAAgLLvLTLIGliKVSYAYqFgLNLTVLQTLhdENidLSPEMSLQEFNddEAYLleDkEKEQRkKVLDSlEKEtaQydqNTliNESladSPeYEtLQqIcFQkKtCpVQLWNtLAqySMVIgFepLQtlQDQMTLEarQTadVgEalEtEewRKVLADQQtRmILCEAwiaqAeaFQdNFESKtTeLMIkNFkTRIpTLFKnkfStLSlgIdLPqsGTFsDkLlTLRdLhHIiIDRNTFrLYtEtLKfIIFgTKgAfGYvCrFsLfEslgEgHRtTEWVfpgQLFpIseGKtTktTaPvTqENgLWLnHmkYEpQYLfgVgiIVFAGlIvWFTaLfqYIVlFGVSgLIvAYFlRKLtLtAFTyVgdLADIEptESgKIStSiAiaGlLaWlAdLAAqfVAlLlEGAdKLCTLnfrLLEgtfVlIvlmSVLGsnMAVSgVRfsLINpKYaaYSlIdDtsLgGlILvsVVsfyfPpVLwLgIIwAILgvALvAgfggLGGLgepLVaaFVlyVvLFLiLlAGSlSAILggLpVsLGIhTgMLvTLAvLgGEgaAlsVGLglTLlAAglLvLtLiGlIKvSYaYqfGLnlTVLQTLhdEnIDLSPEMSLQeFnDdEAYLleDkEKEQdKepaYEEyqNLTylREDkEkeQRkKVLDSlEKEtAQydQnTLINESlAdSPEyEtLQqIcFQkKtCpVQLWNtLAqySMVIgFepLQtlQDQMTLEarQTadVgEalEtEewRKVLADQQtRmILCEAwiaqAeaFQdNFESKtTeLMIkNFkTRIpTLFKnkfStLSlgIdLPqsGTFsDkLlTLRdLhHIiIDRNTFrLYtEtLKfIIFgTKgAfGYvCrFsLfEslgEgHRtTEWVfpgQLFpIseGKtTktTaPvTqENgLWLnHmkYEpQYLfgVgiIVFAGlIvWFTaLfqYIVlFGVSgLIvAYFlRKLtLtAFTyVgdLADIEptESgKIStSiAiaGlLaWlAdLAAqfVAlLlEGAdKLCTLnfrLLEgtfVlIvlmSVLGsnMAVSgVRfsLINpKYaaYSlIdDtsLgGlILvsVVsfyfPpVLwLgIIwAILgvALvAgfggLGGLgepLVaaFVlyVvLFLiLlAGSlSAILggLpVsLGIhTgMLvTLAvLgGEgaAlsVGLglTLlAAglLvLtLiGlIKvSYaYqfGLnlTVLQTLhdEnIDLSPEMSLQeFnDdEAYLleDkEKEQdKepaYEEyqNLTylREDkEkeQRkKVLDSlEKEtAQydQnTLINESlAdSPEyEtLQqIcFQkKtCpVQLWNtLAqySMVIgF"
)

# Drug SMILES strings to test for repurposing
DRUGS_TO_REPURPOSE = {
    "Semaglutide": "CC[C@H](C)[C@@H](C(=O)N[C@@H](C)C(=O)N[C@@H](CC1=CNC2=CC=CC=C21)C(=O)N[C@@H](CC(C)C)C(=O)N[C@@H](C(C)C)C(=O)N[C@@H](CCCNC(=N)N)C(=O)NCC(=O)N[C@@H](CCCNC(=N)N)C(=O)NCC(=O)O)NC(=O)[C@H](CC3=CC=CC=C3)NC(=O)[C@H](CCC(=O)O)NC(=O)[C@H](CCCCNC(=O)COCCOCCNC(=O)COCCOCCNC(=O)CC[C@H](C(=O)O)NC(=O)CCCCCCCCCCCCCCCCC(=O)O)NC(=O)[C@H]",
    "Tirzepatide": "CC[C@H](C)[C@H](N=C(O)[C@H](CC(C)C)N=C(O)[C@H](Cc1c[nH]c2ccccc12)N=C(O)[C@H](CCC(=N)O)N=C(O)[C@@H](N=C(O)[C@H](Cc1ccccc1)N=C(O)[C@H](C)N=C(O)[C@H](CCCCN=C(O)COCCOCCN=C(O)COCCOCCN=C(O)CC[C@H](N=C(O)CCCCCCCCCCCCCCCCCCC(=O)O)C(=O)O)N=C(O)[C@H](CCC(=N)O)N=C(O)[C@H](C)N=C(O)[C@@H](N=C(O)[C@H](CCCCN)N=C(O)[C@H](CC(=O)O)N=C(O)"
}


# Initialize global variables to hold encoding info and loading status
smiles_char_to_int = {}
protein_char_to_int = {}
SMILES_VOCAB_SIZE = 0
PROTEIN_VOCAB_SIZE = 0
is_encoding_ready = False

# --- 2. Load encoding dictionaries and sequence encoding functions ---

print("--- Attempting to load encoding dictionary file ---")
try:
    with open(ENCODING_FILE, 'r') as f:
        encoding_data = json.load(f)
    
    # Extract character-to-index mappings (indices start at 1; 0 reserved for padding)
    smiles_char_to_int = {char: i + 1 for i, char in enumerate(encoding_data['smiles_chars'])}
    protein_char_to_int = {char: i + 1 for i, char in enumerate(encoding_data['protein_chars'])}
    
    # Vocabulary sizes (+1 for padding/placeholder)
    SMILES_VOCAB_SIZE = len(smiles_char_to_int) + 1
    PROTEIN_VOCAB_SIZE = len(protein_char_to_int) + 1

    print(f"✅ Encoding dictionary loaded. SMILES vocab size: {SMILES_VOCAB_SIZE}, Protein vocab size: {PROTEIN_VOCAB_SIZE}")
    is_encoding_ready = True

except FileNotFoundError:
    print(f"❗ Error: Encoding file not found: {ENCODING_FILE}. Please check that the path is: {ENCODING_FILE}")
    
# ... (seq_to_one_hot function unchanged) ...

def seq_to_one_hot(sequence, max_len, char_to_int_dict):
    """
    Convert a sequence (SMILES or protein) to a fixed-length, zero-padded integer sequence.
    """
    sequence = sequence[:max_len]
    encoded_seq = [char_to_int_dict.get(char, 0) for char in sequence]
    
    if len(encoded_seq) < max_len:
        encoded_seq += [0] * (max_len - len(encoded_seq))
    
    return np.array(encoded_seq[:max_len], dtype=np.int32)


def build_deepdta_model(smiles_len, smiles_vocab, protein_len, protein_vocab, embedding_dim=EMBEDDING_DIM):
    """
    Define the Keras architecture for the DeepDTA model.

    ❗ Note: The SMILES CNN block count is set to 4 and the Protein CNN block count is set to 4
    to avoid "negative output dimension" errors caused by insufficient sequence length.
    """
    
    # --- Drug (SMILES) CNN channel ---
    drug_input = Input(shape=(smiles_len,), name='Drug_Input')
    
    drug_x = Embedding(
        input_dim=smiles_vocab,
        output_dim=embedding_dim,
        input_length=smiles_len,
        name='Drug_Embedding'
    )(drug_input)
    
    # SMILES uses 4 Conv-Pool blocks (sequence length 100 constraint)
    print(f"SMILES CNN blocks: 4 (sequence length {smiles_len} constraint)")
    for i in range(4):
        drug_x = Conv1D(filters=32, kernel_size=4, activation='relu', padding='valid')(drug_x)
        drug_x = MaxPool1D(pool_size=2)(drug_x)
    
    drug_output = Flatten(name='Drug_Flatten')(drug_x)

    # --- Target (Protein) CNN channel ---
    target_input = Input(shape=(protein_len,), name='Target_Input')
    
    target_x = Embedding(
        input_dim=protein_vocab,
        output_dim=embedding_dim,
        input_length=protein_len,
        name='Target_Embedding'
    )(target_input)
    
    # Protein now uses 4 Conv-Pool blocks (sequence length 1000 constraint)
    print(f"Protein CNN blocks: 4 (sequence length {protein_len} constraint)")
    for i in range(4): 
        target_x = Conv1D(filters=32, kernel_size=8, activation='relu', padding='valid')(target_x) 
        target_x = MaxPool1D(pool_size=3)(target_x)
        
    target_output = Flatten(name='Target_Flatten')(target_x)
    
    # --- Merge and prediction layers ---
    
    merged = Concatenate()([drug_output, target_output])
    
    merged = Dense(1024, activation='relu')(merged)
    merged = Dense(1024, activation='relu')(merged)
    merged = Dense(512, activation='relu')(merged)
    
    predictions = Dense(1, name='Prediction_Output')(merged)
    
    model = Model(inputs=[drug_input, target_input], outputs=predictions)
    
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mse', 'mae'])
    
    return model

# --- 3. Cross-target prediction function ---

def predict_repurposing_affinity(model, drugs_dict: Dict[str, str], target_seq: str, filename: str = 'best_deepdta_model.keras'):
    """
    Use a trained model to predict affinities of new drugs against a new target.
    """
    if not is_encoding_ready:
        print("❗ Encoding dictionary not loaded; cannot perform prediction.")
        return

    # Attempt to load the model
    try:
        if isinstance(model, str):
            # Assume `model` is the saved model filename
            dta_model = tf.keras.models.load_model(model)
        else:
            dta_model = model
    except Exception as e:
        print(f"❗ Error: Failed to load model {filename}. Please train first or ensure the model file exists. Error: {e}")
        return

    # Encode the CFTR target sequence
    n_drugs = len(drugs_dict)
    target_sequences = [target_seq] * n_drugs
    
    X_target_pred = np.array([
        seq_to_one_hot(s, MAX_PROTEIN_LEN, protein_char_to_int)
        for s in target_sequences
    ])
    
    # Encode drug SMILES sequences
    drug_names = list(drugs_dict.keys())
    drug_smiles = list(drugs_dict.values())
    
    X_drug_pred = np.array([
        seq_to_one_hot(s, MAX_SMILES_LEN, smiles_char_to_int)
        for s in drug_smiles
    ])
    
    print("\n--- 4. Starting cross-target affinity prediction (CFTR) ---")
    
    # Perform prediction
    predictions = dta_model.predict(
        x={'Drug_Input': X_drug_pred, 'Target_Input': X_target_pred}
    )
    
    results = pd.DataFrame({
        "Drug Name": drug_names,
        "Target Name": "CFTR (P13569)",
        "Predicted pIC50": predictions.flatten()
    })

    print(results)
    
    print("\n💡 Interpretation: Higher predicted pIC50 indicates stronger predicted affinity (higher potency).")
    print(f"   This prediction is based on a model trained on EGFR (CHEMBL203) data as an exploratory drug-repurposing test.")


# --- 4. Data preprocessing and model training (main execution block) ---

if __name__ == '__main__':
    if not is_encoding_ready:
        exit() 

    print("\n--- 1. Data loading and preprocessing ---")
    
    data = None
    try:
        data = pd.read_csv(DATASET_FILE)
        data.dropna(subset=['canonical_smiles', 'target_sequence', 'pIC50'], inplace=True)
        print(f"✅ Dataset loaded successfully. {len(data)} valid records found.")
    except FileNotFoundError:
        print(f"❗ Error: Dataset file not found: {DATASET_FILE}. Please check that the path is: {DATASET_FILE}")
    except Exception as e:
        print(f"❗ Error: Unexpected error while loading dataset: {e}")
    
    if data is not None and not data.empty:
        
        # 1.1 Sequence encoding
        print("Encoding drug SMILES and target sequences...")
        X_drug = np.array([
            seq_to_one_hot(s, MAX_SMILES_LEN, smiles_char_to_int)
            for s in data['canonical_smiles']
        ])
        X_target = np.array([
            seq_to_one_hot(s, MAX_PROTEIN_LEN, protein_char_to_int)
            for s in data['target_sequence']
        ])

        # 1.2 Affinity values (pIC50)
        Y = data['pIC50'].values

        # 1.3 Train/test split
        X_drug_train, X_drug_test, X_target_train, X_target_test, Y_train, Y_test = train_test_split(
            X_drug, X_target, Y, test_size=0.2, random_state=42
        )

        print(f"Training samples: {len(Y_train)}, Test samples: {len(Y_test)}")
        
        # --- 2. Model construction and summary ---
        
        print("\n--- 2. Building DeepDTA model (with adjusted convolution block counts) ---")
        dta_model = build_deepdta_model(
            smiles_len=MAX_SMILES_LEN,
            smiles_vocab=SMILES_VOCAB_SIZE,
            protein_len=MAX_PROTEIN_LEN,
            protein_vocab=PROTEIN_VOCAB_SIZE
        )
        
        dta_model.summary()

        # --- 3. Training configuration ---
        
        MODEL_PATH = 'best_deepdta_model.keras'
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='min'),
            ModelCheckpoint(filepath=MODEL_PATH, monitor='val_loss', save_best_only=True, verbose=1)
        ]
        
        BATCH_SIZE = 128
        # EPOCHS = 100 # Set appropriately for real training

        print("\n--- 3. Model training/loading step ---")
        
        # Train the model
        print("Starting model training...")
        history = dta_model.fit(
             x={'Drug_Input': X_drug_train, 'Target_Input': X_target_train},
             y=Y_train,
             batch_size=BATCH_SIZE,
             epochs=100, # example
             validation_data=({'Drug_Input': X_drug_test, 'Target_Input': X_target_test}, Y_test),
             callbacks=callbacks,
             verbose=2
        )
        print(f"Training finished. Best model saved to {MODEL_PATH}")

        # --- 4. Cross-target drug repurposing prediction ---

        predict_repurposing_affinity(
            model=MODEL_PATH,
            drugs_dict=DRUGS_TO_REPURPOSE,
            target_seq=CFTR_SEQUENCE,
            filename=MODEL_PATH
         )
        
    else:
        print("❗ Cannot proceed: dataset failed to load or is empty. Skipping model building and training.")
