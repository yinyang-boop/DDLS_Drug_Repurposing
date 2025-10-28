import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import numpy as np
import json
import logging
from pathlib import Path
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr
import time
from typing import Tuple, Optional, Dict

# --- Configuration ---
PROCESSED_DIR = Path("data/processed")
DATA_PATH = PROCESSED_DIR / "deepdta_dataset.csv"
ENCODING_PATH = PROCESSED_DIR / "deepdta_encoding.json"
MODEL_SAVE_PATH = Path("models/deepdta_egfr_best.pt")
MODEL_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# =======================================================================
# I. Model Architecture (BaselineDeepDTA)
# =======================================================================

class BaselineDeepDTA(nn.Module):
    """
    Baseline Y-shaped DeepDTA architecture.
    Uses CNNs to process protein sequences and compound SMILES separately,
    then fuses features for prediction.
    """
    
    def __init__(self, 
                 protein_vocab_size: int = 25, 
                 compound_vocab_size: int = 64, 
                 protein_embed_dim: int = 128,
                 compound_embed_dim: int = 128,
                 protein_max_len: int = 1000,
                 compound_max_len: int = 200,
                 hidden_dims: list = [256, 128, 64],
                 dropout: float = 0.2):
        super(BaselineDeepDTA, self).__init__()
        
        # Protein branch
        self.protein_embedding = nn.Embedding(protein_vocab_size, protein_embed_dim)
        self.protein_cnn = nn.Sequential(
            nn.Conv1d(protein_embed_dim, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(64, 96, kernel_size=7, padding=3),
            nn.ReLU(),
            # Ensure fixed dimension after global pooling
            nn.AdaptiveMaxPool1d(1) 
        )
        
        # Compound SMILES branch
        self.compound_embedding = nn.Embedding(compound_vocab_size, compound_embed_dim)
        self.compound_cnn = nn.Sequential(
            nn.Conv1d(compound_embed_dim, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(64, 96, kernel_size=7, padding=3),
            nn.ReLU(),
            # Ensure fixed dimension after global pooling
            nn.AdaptiveMaxPool1d(1)
        )
        
        # Fusion and prediction layers
        self.fusion_layers = nn.Sequential()
        # 96 (protein) + 96 (compound) = 192
        input_dim = 96 + 96 
        
        for i, hidden_dim in enumerate(hidden_dims):
            self.fusion_layers.add_module(f'fc_{i}', nn.Linear(input_dim, hidden_dim))
            self.fusion_layers.add_module(f'relu_{i}', nn.ReLU())
            self.fusion_layers.add_module(f'dropout_{i}', nn.Dropout(dropout))
            input_dim = hidden_dim
        
        # Output layer: regression predicting binding affinity (e.g., pIC50)
        self.output_layer = nn.Linear(hidden_dims[-1], 1)
        
    def forward(self, protein_seq: torch.Tensor, compound_seq: torch.Tensor) -> torch.Tensor:
        # Protein forward pass
        # (batch, seq_len) -> (batch, seq_len, embed_dim)
        protein_embedded = self.protein_embedding(protein_seq) 
        # (batch, embed_dim, seq_len) - Conv1d expects this format
        protein_embedded = protein_embedded.transpose(1, 2) 
        protein_features = self.protein_cnn(protein_embedded).squeeze(-1) # (batch, 96)
        
        # Compound forward pass
        compound_embedded = self.compound_embedding(compound_seq)
        compound_embedded = compound_embedded.transpose(1, 2)
        compound_features = self.compound_cnn(compound_embedded).squeeze(-1) # (batch, 96)
        
        # Feature fusion
        combined_features = torch.cat([protein_features, compound_features], dim=1)
        
        # Through fully connected layers
        prediction = self.fusion_layers(combined_features)
        
        # Final prediction
        prediction = self.output_layer(prediction)
        return prediction

# =======================================================================
# II. Dataset and Preprocessing
# =======================================================================

def seq_to_tensor(sequence: str, char_dict: Dict[str, int], max_len: int) -> torch.Tensor:
    """
    Encode a sequence as an integer tensor with padding/truncation.
    0 is used for padding.
    """
    encoded = [char_dict.get(c, 0) for c in sequence]
    
    if len(encoded) > max_len:
        # Truncate
        encoded = encoded[:max_len]
    elif len(encoded) < max_len:
        # Pad with zeros at the end
        padding_needed = max_len - len(encoded)
        encoded.extend([0] * padding_needed)
        
    return torch.tensor(encoded, dtype=torch.long)

class DeepDTADataset(Dataset):
    """Custom PyTorch dataset for DeepDTA"""
    def __init__(self, df: pd.DataFrame, encoding_data: Dict):
        self.df = df
        self.smiles_dict = encoding_data['smiles_dict']
        self.protein_dict = encoding_data['protein_dict']
        self.max_smiles_len = encoding_data['max_smiles_len']
        self.max_protein_len = encoding_data['max_protein_len']

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        smiles = row['canonical_smiles']
        protein = row['target_sequence']
        pIC50 = row['pIC50']
        
        # Encode SMILES and protein sequence
        compound_tensor = seq_to_tensor(smiles, self.smiles_dict, self.max_smiles_len)
        protein_tensor = seq_to_tensor(protein, self.protein_dict, self.max_protein_len)
        
        # pIC50 as float tensor
        pIC50_tensor = torch.tensor(pIC50, dtype=torch.float32).view(1)
        
        return protein_tensor, compound_tensor, pIC50_tensor

# =======================================================================
# III. Training Loop
# =======================================================================

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
    """Compute MSE and Pearson correlation coefficient"""
    mse = np.mean((y_true - y_pred) ** 2)
    # Ensure at least two points to compute correlation
    if len(y_true) > 1:
        pearson, _ = pearsonr(y_true, y_pred)
    else:
        pearson = np.nan
    return mse, pearson

def train_model(epochs: int = 100, batch_size: int = 128, lr: float = 1e-4):
    """
    Train, validate, and test the DeepDTA model.
    """
    # 1. Check CUDA device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # 2. Load data and encodings
    try:
        df = pd.read_csv(DATA_PATH)
        with open(ENCODING_PATH, 'r') as f:
            encoding_data = json.load(f)
    except FileNotFoundError:
        logger.error(f"Data file {DATA_PATH} or encoding file {ENCODING_PATH} not found. Please run data_pipeline_auto.py first.")
        return
    
    # 3. Prepare model parameters
    smiles_vocab_size = len(encoding_data['smiles_dict']) + 1
    protein_vocab_size = len(encoding_data['protein_dict']) + 1
    max_smiles_len = encoding_data['max_smiles_len']
    max_protein_len = encoding_data['max_protein_len']

    # 4. Initialize model, loss, and optimizer
    model = BaselineDeepDTA(
        protein_vocab_size=protein_vocab_size, 
        compound_vocab_size=smiles_vocab_size,
        protein_max_len=max_protein_len,
        compound_max_len=max_smiles_len
    ).to(device)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    logger.info(f"Model parameters initialized:")
    logger.info(f"  - Compound vocab size: {smiles_vocab_size}, max length: {max_smiles_len}")
    logger.info(f"  - Protein vocab size: {protein_vocab_size}, max length: {max_protein_len}")
    logger.info(f"  - Total samples: {len(df)}")
    
    # 5. Split datasets
    full_dataset = DeepDTADataset(df, encoding_data)
    
    # 80% train, 10% val, 10% test
    train_size = int(0.8 * len(full_dataset))
    val_size = int(0.1 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    logger.info(f"Datasets split (train: {train_size}, val: {val_size}, test: {test_size})")

    # 6. Training loop
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        start_time = time.time()
        
        # --- Training phase ---
        model.train()
        train_loss = 0
        
        for protein_seq, compound_seq, pIC50 in train_loader:
            protein_seq, compound_seq, pIC50 = protein_seq.to(device), compound_seq.to(device), pIC50.to(device)
            
            optimizer.zero_grad()
            output = model(protein_seq, compound_seq)
            loss = criterion(output, pIC50)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * protein_seq.size(0)
            
        train_loss /= len(train_loader.dataset)
        
        # --- Validation phase ---
        model.eval()
        val_loss = 0
        val_preds, val_targets = [], []
        
        with torch.no_grad():
            for protein_seq, compound_seq, pIC50 in val_loader:
                protein_seq, compound_seq, pIC50 = protein_seq.to(device), compound_seq.to(device), pIC50.to(device)
                
                output = model(protein_seq, compound_seq)
                loss = criterion(output, pIC50)
                
                val_loss += loss.item() * protein_seq.size(0)
                val_preds.extend(output.cpu().numpy().flatten())
                val_targets.extend(pIC50.cpu().numpy().flatten())
                
        val_loss /= len(val_loader.dataset)
        val_mse, val_pearson = calculate_metrics(np.array(val_targets), np.array(val_preds))
        
        end_time = time.time()

        logger.info(
            f"Epoch {epoch+1}/{epochs} | "
            f"Time: {end_time - start_time:.2f}s | "
            f"Train MSE: {train_loss:.4f} | "
            f"Val MSE: {val_loss:.4f} | "
            f"Val Pearson: {val_pearson:.4f}"
        )
        
        # 7. Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            logger.info(f"**Model saved to {MODEL_SAVE_PATH} (val loss: {best_val_loss:.4f})**")

    # 8. Final testing
    logger.info("-" * 50)
    logger.info("Starting final testing...")
    
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    model.eval()
    
    test_preds, test_targets = [], []
    with torch.no_grad():
        for protein_seq, compound_seq, pIC50 in test_loader:
            protein_seq, compound_seq, pIC50 = protein_seq.to(device), compound_seq.to(device), pIC50.to(device)
            
            output = model(protein_seq, compound_seq)
            test_preds.extend(output.cpu().numpy().flatten())
            test_targets.extend(pIC50.cpu().numpy().flatten())

    test_mse, test_pearson = calculate_metrics(np.array(test_targets), np.array(test_preds))
    
    logger.info("--- Final test results ---")
    logger.info(f"Test MSE: {test_mse:.4f}")
    logger.info(f"Test Pearson correlation: {test_pearson:.4f}")
    logger.info("DeepDTA model training completed.")


if __name__ == "__main__":
    # Default: train 100 epochs with learning rate 1e-4
    train_model(epochs=100, batch_size=256, lr=1e-4)
