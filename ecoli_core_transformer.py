import os
import gc
import random
import time
from datetime import date

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F

from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import r2_score, mean_absolute_error

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
import seaborn as sns

DATA_PATH = "./data/2025-07-15_full_training_data_98066_samples.csv" # carbons log-uniform, others uniform

# Set all random seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def print_gpu_memory():
    """
    Displays the current GPU memory usage by PyTorch on the active CUDA device.
    Useful for monitoring GPU memory usage during training.
    """
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        allocated = torch.cuda.memory_allocated(device) / 1024**2
        reserved = torch.cuda.memory_reserved(device) / 1024**2
        print(f'Allocated memory: {allocated:.2f} MB')
        print(f'Reserved memory: {reserved:.2f} MB')
    else:
        print("CUDA is not available.")

class AttentionBlock(nn.Module):
    """Multi-head attention block with per-head diffusion of c."""
    def __init__(self, d_model=128, n_heads=8, dropout=0.05):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads

        self.layer_norm = nn.LayerNorm(d_model)

        self.mha = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Learnable head aggregation
        self.head_scores = nn.Parameter(torch.zeros(n_heads))

    def forward(self, x, c):
        """
        x: (B, S, d_model)
        c: (B, S, 1)
        returns: x_out (B, S, d_model), c_out (B, S, 1)
        """
        x_norm = self.layer_norm(x)

        # Get per-head attention weights: (B, H, S, S)
        attn_out, attn_weights = self.mha(
            x_norm, x_norm, x_norm,
            need_weights=True,
            average_attn_weights=False
        )
        
        x_out = attn_out + x

        # Per-head diffusion of c:
        # (B, H, S, S) @ (B, 1, S, 1) -> (B, H, S, 1)
        c_heads = torch.matmul(attn_weights, c.unsqueeze(1))
        alpha = F.softmax(self.head_scores, dim=0).view(1, self.n_heads, 1, 1)  # (1,H,1,1)
        c_att = (c_heads * alpha).sum(dim=1)  # (B, S, 1)

        c_out = c_att + c

        return x_out, c_out

class FeedForwardBlock(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.05):
        super().__init__()

        self.d_model = d_model + 1
        self.d_ff = d_ff

        self.layer_norm = nn.LayerNorm(self.d_model)
        self.linear1 = nn.Linear(self.d_model, self.d_ff)
        self.activation = nn.GELU()
        self.linear2 = nn.Linear(self.d_ff, self.d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, c):
        y = torch.cat((x, c), dim=2)
        
        norm_y = self.layer_norm(y)
        hidden = self.linear1(norm_y)
        hidden = self.activation(hidden)
        hidden = self.dropout(hidden)
        output = self.linear2(hidden)

        return output + y
    
class FluxTransformerLayer(nn.Module):
    """Single transformer block without embedding layer"""
    def __init__(self, d_model=128, n_heads=8, d_ff=640, dropout=0.05):
        super().__init__()
        self.d_model = d_model
        
        self.attention_block = AttentionBlock(d_model, n_heads, dropout)
        self.feedforward_block = FeedForwardBlock(d_model, d_ff, dropout)
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
    
    def forward(self, x, c):
        attn_x, attn_c = self.attention_block(x, c)
        ff_output = self.feedforward_block(attn_x, attn_c)
        
        # Split the concatenated output
        updated_x = ff_output[:, :, :-1]
        updated_c = ff_output[:, :, -1:]
        
        return updated_x, updated_c
    
class FluxTransformer(nn.Module):
    def __init__(
        self,
        vocab_size=115,
        d_model=128,
        n_heads=8,
        n_layers=3,
        d_ff=640,
        dropout=0.05
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        
        self.input_embedding = nn.Embedding(vocab_size, d_model)
        
        self.layers = nn.ModuleList([
            FluxTransformerLayer(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                dropout=dropout
            )
            for _ in range(n_layers)
        ])

    def forward(self, c, return_embedding=False):
        batch_size = c.size(0)
        
        # Create token indices once
        y = torch.arange(self.vocab_size, device=c.device)
        y = y.unsqueeze(0).expand(batch_size, -1)  # (batch, seq)
        
        # Embed tokens once
        x = self.input_embedding(y)  # (batch, seq, d_model)
        
        for layer in self.layers:
            x, c = layer(x, c)

        if return_embedding:
            return x  # Return embeddings (batch, seq, d_model)
        
        return c
    
def load_data(filepath):
    """
    Load and preprocess metabolic flux training data
    
    Parameters:
        filepath (str): Path to the CSV file.

    Returns:
        X_combined (ndarray): Combined input matrix (inputs in the first columns, outputs are zeros).
        y_combined (ndarray): Combined output matrix (inputs are zeros, outputs in the last columns).
        all_columns (list): List of column names (input + output).
    """
    inputs = [
        'EX_glc__D_e', 'EX_fru_e', 'EX_lac__D_e', 'EX_pyr_e', 'EX_ac_e', 'EX_akg_e', 
        'EX_succ_e', 'EX_fum_e', 'EX_mal__L_e', 'EX_etoh_e', 'EX_acald_e', 'EX_for_e',
        'EX_gln__L_e', 'EX_glu__L_e',
        'EX_co2_e', 'EX_h_e', 'EX_h2o_e', 'EX_nh4_e', 'EX_o2_e', 'EX_pi_e',
    ]

    outputs = [
        'ACALD_flux', 'ACALDt_flux', 'ACKr_flux', 'ACONTa_flux', 'ACONTb_flux',
        'ACt2r_flux', 'ADK1_flux', 'AKGDH_flux', 'AKGt2r_flux', 'ALCD2x_flux',
        'ATPM_flux', 'ATPS4r_flux', 'Biomass_Ecoli_core_flux', 'CO2t_flux', 'CS_flux',
        'CYTBD_flux', 'D_LACt2_flux', 'ENO_flux', 'ETOHt2r_flux', 'EX_ac_e_flux',
        'EX_acald_e_flux', 'EX_akg_e_flux', 'EX_co2_e_flux', 'EX_etoh_e_flux', 'EX_for_e_flux',
        'EX_fru_e_flux', 'EX_fum_e_flux', 'EX_glc__D_e_flux', 'EX_gln__L_e_flux', 'EX_glu__L_e_flux',
        'EX_h_e_flux', 'EX_h2o_e_flux', 'EX_lac__D_e_flux', 'EX_mal__L_e_flux', 'EX_nh4_e_flux',
        'EX_o2_e_flux', 'EX_pi_e_flux', 'EX_pyr_e_flux', 'EX_succ_e_flux', 'FBA_flux',
        'FBP_flux', 'FORt2_flux', 'FORti_flux', 'FRD7_flux', 'FRUpts2_flux',
        'FUM_flux', 'FUMt2_2_flux', 'G6PDH2r_flux', 'GAPD_flux', 'GLCpts_flux',
        'GLNS_flux', 'GLNabc_flux', 'GLUDy_flux', 'GLUN_flux', 'GLUSy_flux',
        'GLUt2r_flux', 'GND_flux', 'H2Ot_flux', 'ICDHyr_flux', 'ICL_flux',
        'LDH_D_flux', 'MALS_flux', 'MALt2_2_flux', 'MDH_flux', 'ME1_flux',
        'ME2_flux', 'NADH16_flux', 'NADTRHD_flux', 'NH4t_flux', 'O2t_flux',
        'PDH_flux', 'PFK_flux', 'PFL_flux', 'PGI_flux', 'PGK_flux',
        'PGL_flux', 'PGM_flux', 'PIt2r_flux', 'PPC_flux', 'PPCK_flux',
        'PPS_flux', 'PTAr_flux', 'PYK_flux', 'PYRt2_flux', 'RPE_flux',
        'RPI_flux', 'SUCCt2_2_flux', 'SUCCt3_flux', 'SUCDi_flux', 'SUCOAS_flux',
        'TALA_flux', 'THD2_flux', 'TKT1_flux', 'TKT2_flux', 'TPI_flux'
    ]
    
    df = pd.read_csv(filepath)

    # Fill missing inputs with 0 (i.e., not uptaken)
    df[inputs] = df[inputs].fillna(0)

    print(f"\nLoaded data with {len(df)} samples from {filepath}")
    print(f"Number of input features: {len(inputs)}")
    print(f"Number of output targets: {len(outputs)}")

    X = df[inputs].to_numpy(dtype=np.float32)
    y = df[outputs].to_numpy(dtype=np.float32)

    X_combined = np.hstack([X, np.zeros_like(y)])
    y_combined = np.hstack([np.zeros_like(X), y])

    return X_combined, y_combined, inputs, outputs

def prepare_tensors(X, y, test_size=0.2, device="cpu"):
    """
    Split data into train/test and convert to PyTorch tensors.
    
    Parameters:
        X (ndarray): Input features.
        y (ndarray): Output targets.
        test_size (float): Fraction of data to reserve for testing.
        device (str or torch.device): Device to move tensors to.
    
    Returns:
        X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor (torch.Tensor)
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")

    # Optional: standardize inputs (only input features, i.e., first 20 columns)
    # scaler = StandardScaler()
    # X_train[:, :20] = scaler.fit_transform(X_train[:, :20])
    # X_test[:, :20] = scaler.transform(X_test[:, :20])

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(device)

    return X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor

def create_dataloaders(X_train, y_train, X_test, y_test, batch_size):
    """
    Create PyTorch DataLoaders for training and testing.

    Parameters:
        X_train, y_train (Tensor): Training data and labels.
        X_test, y_test (Tensor): Test data and labels.
        batch_size (int): Batch size for loading.

    Returns:
        train_loader, test_loader (DataLoader): PyTorch DataLoaders.
    """
    train_dataset = TensorDataset(X_train.unsqueeze(-1), y_train.unsqueeze(-1))
    test_dataset = TensorDataset(X_test.unsqueeze(-1), y_test.unsqueeze(-1))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

def train_model(d_model=128, n_heads=8, n_layers=3, d_ff=640, num_epochs=1000, learning_rate=0.001, dropout=0.05):
    start_time = time.time()

    model = FluxTransformer(
        vocab_size=115,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_ff,
        dropout=dropout
    ).to(device)
    
    #optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    #optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.98),
        eps=1e-9,
        weight_decay=1e-4
    )
    #criterion = nn.MSELoss()
    criterion = nn.HuberLoss()

    best_test_loss = float('inf')
    best_epoch = -1

    train_losses = []
    test_losses = []

    for epoch in range(num_epochs):
        model.train()
        epoch_train_loss = 0.0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_train_loss += loss.item() * batch_X.size(0)

            # Explicitly free tensors
            del predictions, loss
            torch.cuda.empty_cache()
        
        epoch_train_loss /= len(train_loader.dataset)
        train_losses.append(epoch_train_loss)

        # Evaluation
        model.eval()
        epoch_test_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                predictions = model(batch_X)
                loss = criterion(predictions, batch_y)
                epoch_test_loss += loss.item() * batch_X.size(0)

                # Explicitly free tensors
                del predictions, loss
                torch.cuda.empty_cache()
        
        epoch_test_loss /= len(test_loader.dataset)
        test_losses.append(epoch_test_loss)

        if (epoch+1) % 2 == 0:
            print(f"Epoch {epoch+1}/{num_epochs} | "
                f"Train Loss: {epoch_train_loss:.6f} | "
                f"Test Loss: {epoch_test_loss:.6f}")
        
        if epoch_test_loss < best_test_loss:
            best_test_loss = epoch_test_loss
            best_epoch = epoch + 1
        
        # Additional memory cleanup after epoch
        torch.cuda.empty_cache()
        gc.collect()

    print('Training Completed.')
    end_time = time.time()
    elapsed_time = end_time - start_time
    mins, secs = divmod(elapsed_time, 60)
    print(f"Training took {int(mins)} min {secs:.1f} sec.")
    print(f"Best test loss: {best_test_loss:.6f} at epoch {best_epoch}")
    
    return train_losses, test_losses, model, optimizer

def plot_loss_curves(train_losses, test_losses, d_model, n_heads, n_layers, d_ff, save_path=None, log_scale=True):
    plt.figure(figsize=(14, 10))
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.plot(train_losses, label="Training Loss")
    plt.plot(test_losses, label="Test Loss")
    if log_scale:
        plt.yscale('log')
    plt.xlabel("Epoch", fontsize=18)
    plt.ylabel("Loss", fontsize=18)
    plt.title(f"Loss Curves (d={d_model}, h={n_heads}, l={n_layers}, ff={d_ff})", fontsize=20)
    plt.grid(True)
    plt.legend(fontsize=16)
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"\nTraining curve saved to {save_path}")
        plt.close()
    else:
        plt.show()

def plot_diagnostics_2x2(y_true, y_pred, label, save_path):
    """Creates a 2x2 matrix of plots: true vs predicted, residuals, error distribution, and histogram of actuals"""
    if np.std(y_true) < 1e-6:
        print(f"Skipping {label} due to near-constant target values.")
        return
    residuals = y_true - y_pred

    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    
    # True vs Predicted
    axs[0, 0].scatter(y_true, y_pred, alpha=0.2, s=5, color='royalblue')
    axs[0, 0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'k--', lw=2)
    axs[0, 0].set_title(f'True vs Predicted: {label}')
    axs[0, 0].set_xlabel('True value')
    axs[0, 0].set_ylabel('Predicted')
    axs[0, 0].grid(True)

    # Residuals plot
    axs[0, 1].scatter(y_true, residuals, alpha=0.15, color='darkorange')
    axs[0, 1].axhline(y=0, color='r', linestyle='-')
    axs[0, 1].set_title(f'Residuals: {label}')
    axs[0, 1].set_xlabel('True value')
    axs[0, 1].set_ylabel('Residuals')
    axs[0, 1].grid(True)

    # Error distribution
    sns.histplot(residuals.ravel(), kde=True, ax=axs[1, 0], legend=False, color='indianred')
    axs[1, 0].set_title(f'Prediction Error Distribution: {label}')
    axs[1, 0].set_xlabel('Prediction Error')
    axs[1, 0].set_ylabel('Frequency')
    axs[1, 0].grid(True)

    # Histogram of actual values
    sns.histplot(y_true.ravel(), kde=True, bins=100, ax=axs[1, 1], legend=False, color='mediumseagreen')
    axs[1, 1].set_title(f'True Value Distribution: {label}')
    axs[1, 1].set_xlabel('True Value')
    axs[1, 1].set_ylabel('Frequency')
    axs[1, 1].grid(True)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()

def plot_prediction_sample(X_test, y_test, model, save_path=None):
    X_test_ = X_test.unsqueeze(-1)
    y_test_ = y_test.unsqueeze(-1)
    j = np.random.randint(0, X_test_.size(0), 1)[0]
    pred = model(X_test_[j].unsqueeze(0))
    true = y_test_[j].unsqueeze(0)
    plt.plot(pred[0,:,0].cpu().detach().numpy(), label='Predicted')
    plt.plot(true[0,:,0].cpu().detach().numpy(), label='True')

    # add a red vertical line at index 32 (biomass)
    plt.axvline(x=32, color='red', linestyle='--', linewidth=1)
    
    plt.legend()
    plt.title(f"Random sample: index {j}")
    plt.xlabel("Reaction ID")
    plt.ylabel("Rate")
    plt.grid(True)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Graph for prediction samples saved to {save_path}")
        plt.close()
    else:
        plt.show()

def plot_pre_attention_embeddings_tsne(model, output_cols, perplexity=30.0, save_path=None):
    """
    Visualize the model's fundamental token embeddings (pre-attention) using t-SNE.
    Shows the initial learned representations before any context is applied.
    
    Parameters:
        model: Trained FluxTransformer model
        output_cols: List of output column names (for labeling)
        perplexity: t-SNE perplexity parameter
        save_path: Directory to save plot
    """
    device = next(model.parameters()).device
    n_outputs = len(output_cols)
    
    # Get the raw embeddings from the model's embedding layer
    with torch.no_grad():
        token_ids = torch.arange(20, 20+n_outputs, device=device)
        embeddings = model.input_embedding(token_ids).cpu().numpy()

    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)

    colors = plt.cm.tab20(np.linspace(0, 1, n_outputs)) 
    #colors = plt.cm.gist_rainbow(np.linspace(0, 1, n_outputs))
    #colors = plt.cm.coolwarm(np.linspace(0, 1, n_outputs))

    plt.figure(figsize=(14, 10))
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=colors, s=100, alpha=1.0)

    for i, flux in enumerate(output_cols):
        if flux in output_cols:
            clean_flux = flux.replace('_flux', '')
            plt.annotate(clean_flux, (embeddings_2d[i, 0], embeddings_2d[i, 1]), 
                         xytext=(5, 5), textcoords='offset points',
                         fontsize=9, bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7))

    plt.title('t-SNE of Fundamental Output Token Embeddings\n(Pre-attention, context-independent)', fontsize=20)
    plt.xlabel('t-SNE 1', fontsize=18)
    plt.ylabel('t-SNE 2', fontsize=18)
    plt.grid(alpha=0.2)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
        print(f"Pre attention embedding visualization saved to {save_path}")
    else:
        plt.show()

def plot_post_attention_zero_context_tsne(model, output_cols, perplexity=30.0, save_path=None):
    """
    Visualize token embeddings after attention layers with zero context using t-SNE.
    Shows how the model transforms embeddings without metabolic context.
    
    Parameters:
        model: Trained FluxTransformer model
        output_cols: List of output column names (for labeling)
        perplexity: t-SNE perplexity parameter
        save_path: Directory to save plot
    """
    device = next(model.parameters()).device
    n_outputs = len(output_cols)

    # Get all embeddings
    model.eval()
    with torch.no_grad():
        # Create zero c: (batch_size=1, seq_len=115, 1)
        dummy_c = torch.zeros(1, model.vocab_size, 1, device=device)
        # Get embeddings (shape: [1, 115, d_model])
        embeddings = model(dummy_c, return_embedding=True)  
        # Extract output tokens (indices 20-114) and remove batch dimension
        output_embeddings = embeddings[0, 20:20+n_outputs, :].cpu().numpy()  # shape: (95, d_model)

    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    embeddings_2d = tsne.fit_transform(output_embeddings)

    colors = plt.cm.tab20(np.linspace(0, 1, n_outputs)) 
    #colors = plt.cm.coolwarm(np.linspace(0, 1, n_outputs))

    plt.figure(figsize=(14, 10))
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=colors, s=100, alpha=1.0)

    for i, flux in enumerate(output_cols):
        if flux in output_cols:
            clean_flux = flux.replace('_flux', '')
            plt.annotate(clean_flux, (embeddings_2d[i, 0], embeddings_2d[i, 1]), 
                         xytext=(5, 5), textcoords='offset points',
                         fontsize=9, bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7))
            
    plt.title('t-SNE of Output Flux Embeddings, Zero Context\n(Each flux colored individually)', fontsize=20)
    plt.xlabel('t-SNE 1', fontsize=18)
    plt.ylabel('t-SNE 2', fontsize=18)
    plt.grid(alpha=0.2)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
        print(f"Post attention, zero c t-SNE plot for output tokens saved to {save_path}")
    else:
        plt.show()

def plot_post_attention_real_context(
    model, 
    output_cols, 
    X_test, 
    method='pca',
    n_components=2,
    n_samples=1000, 
    perplexity=30.0, # t-SNE parameter
    n_neighbors=15,  # UMAP parsmeter
    min_dist=0.1, # UMAP parameter
    save_path=None
):
    """
    Visualize token embeddings from real metabolic contexts using PCA or t-SNE.
    Shows how input conditions affect the model's internal representations.
    
    Args:
        model: Trained model
        output_cols: List of output flux names
        X_test: Input tensor for test data
        method: 'pca' or 'tsne' (default: 'pca')
        n_components: number of dimensions for visualization
        n_samples: Number of samples to visualize
        perplexity: t-SNE perplexity (only for t-SNE method)
        n_neighbors: UMAP parameter controlling neighborhood size
        min_dist: UMAP parameter controlling cluster spacing
        save_path: Optional path to save the figure
    """
    device = next(model.parameters()).device
    n_outputs = len(output_cols)
    n_samples = min(n_samples, len(X_test))
    
    model.eval()
    all_embeddings = []
    flux_indices = []
    
    with torch.no_grad():
        batch_size = 128
        for i in range(0, n_samples, batch_size):
            batch = X_test[i:i+batch_size, :20].unsqueeze(-1).to(device)
            batch_context = torch.zeros(batch.size(0), model.vocab_size, 1, device=device)
            batch_context[:, :20] = batch
            
            embeddings = model(batch_context, return_embedding=True)
            output_embeddings = embeddings[:, 20:20+n_outputs, :]
            
            all_embeddings.append(output_embeddings.cpu())
            flux_indices.extend([np.arange(n_outputs)] * len(batch))
    
    # Combine all samples [total_samples, n_outputs, d_model]
    all_embeddings = torch.cat(all_embeddings, dim=0)
    flux_indices = np.concatenate(flux_indices)
    
    # Flatten embeddings [n_points, d_model]
    embeddings_flat = all_embeddings.reshape(-1, all_embeddings.shape[-1]).numpy()
    
    # Dimensionality reduction method
    if method.lower() == 'pca':
        reducer = PCA(n_components=n_components)
        method_name = 'PCA'
    elif method.lower() == 'tsne':
        reducer = TSNE(n_components=n_components, perplexity=perplexity, random_state=42)
        method_name = 't-SNE'
    elif method.lower() == 'umap':
        try:
            from umap import UMAP
            reducer = UMAP(n_components=n_components, 
                          n_neighbors=n_neighbors, 
                          min_dist=min_dist, 
                          random_state=42)
            method_name = 'UMAP'
        except ImportError:
            raise ImportError("UMAP not installed. Please install umap-learn: pip install umap-learn")
    else:
        raise ValueError(f"Invalid method: '{method}'. Choose 'pca' or 'tsne'")
    
    embeddings_rd = reducer.fit_transform(embeddings_flat)

    # Create colormap and plot
    colors = plt.cm.nipy_spectral(np.linspace(0, 1, n_outputs))
    discrete_cmap = ListedColormap(colors)

    # Calculate cluster centers
    centers = []
    for flux_idx in range(n_outputs):
        mask = flux_indices == flux_idx
        if mask.any():
            center = np.median(embeddings_rd[mask], axis=0)
            centers.append((output_cols[flux_idx], center))
    
    if n_components == 3:
        fig = plt.figure(figsize=(16, 12))
        ax = fig.add_subplot(111, projection='3d')
        
        sc = ax.scatter(
            embeddings_rd[:, 0], 
            embeddings_rd[:, 1], 
            embeddings_rd[:, 2],
            c=flux_indices, 
            cmap=discrete_cmap, 
            alpha=0.15, 
            s=7,
            edgecolors='none'
        )
                
        ax.set_title(f'Post-Attention Embeddings\n({n_samples} input conditions, {method_name})', fontsize=16)
        ax.set_xlabel(f'{method_name} 1', fontsize=12)
        ax.set_ylabel(f'{method_name} 2', fontsize=12)
        ax.set_zlabel(f'{method_name} 3', fontsize=12)
        ax.grid(alpha=0.2)
        
    elif n_components == 2:
        plt.figure(figsize=(16, 12))
        plt.scatter(
            embeddings_rd[:, 0], embeddings_rd[:, 1],
            c=flux_indices, cmap=discrete_cmap, alpha=0.2, s=10,
            edgecolors='none'
        )
        
        # Plot and annotate cluster centers
        for flux, center in centers:
            clean_flux = flux.replace('_flux', '')
            plt.scatter(
                center[0], center[1],
                s=20, marker='o',
                color=colors[output_cols.index(flux) % len(colors)],
                edgecolor='black', linewidth=0.5
            )
            plt.annotate(
                clean_flux, center,
                xytext=(5, 5), textcoords='offset points',
                fontsize=9,
                bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.8)
            )
        
        plt.title(f'Post-Attention Embeddings\n({n_samples} input conditions, {method_name})', fontsize=16)
        plt.xlabel(f'{method_name} 1', fontsize=12)
        plt.ylabel(f'{method_name} 2', fontsize=12)
        plt.grid(alpha=0.1)
    
    else:
        raise ValueError("n_components must be 2 or 3")
    
    # Save or display the plot
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
        print(f"Plot saved to {save_path}")
    else:
        plt.show()

def plot_post_attention_grouped_tsne(model, output_cols, X_test, n_samples=1000, 
                                         perplexity=30.0, save_path=None):
    """
    Visualize token embeddings from real metabolic contexts with flux grouping.
    Colors represent metabolic pathway groups.
    """
    device = next(model.parameters()).device
    n_outputs = len(output_cols)
    n_samples = min(n_samples, len(X_test))
    
    clean_fluxes = [f.replace('_flux', '') for f in output_cols]

    group_colors = {
        # Fermentation (red/orange)
        'Ethanol Fermentation': "#FB737E",
        'Acetate Production': "#FEABA5",
        'Lactate Production': '#FFDAC1',
        'Formate Production': "#F9AA61",
        
        # Central Carbon Metabolism (greens)
        'Glycolysis': '#6CC070',
        'Gluconeogenesis': "#7CECCA",
        'PP Pathway': '#C7F0BD',
        
        # TCA & Associated Pathways (blues)
        'TCA Cycle': "#237DE4",
        'Glyoxylate Shunt': '#63B4FF',
        'Anaplerotic': "#B0DEF4",
        
        # Respiration & Energy (purples)
        'OxPhos': "#A4409F",
        'ATP Maintenance': '#C77DFF',
        'Anaerobic Respiration': "#BBACD7",
        
        # Nitrogen Metabolism (yellows/goldens)
        'Nitrogen Uptake': "#E2C000",
        'Glu/Gln Synthesis': "#FFF1A8",
        
        # Transport & Exchange (gray)
        'Transport & EX': '#B0B0B0',
        
        # Biomass (distinctive color)
        'Biomass': "#483954"
    }
    
    # Define metabolic pathway groups
    pathway_groups = {
        # Fermentation Pathways
        'Ethanol Fermentation': ['ACALD', 'ALCD2x', 'ETOHt2r', 'EX_etoh_e', 'ACALDt', 'EX_acald_e'],
        'Acetate Production': ['PTAr', 'ACKr', 'ACt2r', 'EX_ac_e'],
        'Lactate Production': ['LDH_D', 'D_LACt2', 'EX_lac__D_e'],
        'Formate Production': ['PFL', 'FORt2', 'FORti', 'EX_for_e'],
        
        # Central Carbon Metabolism
        'Glycolysis': ['GLCpts', 'FRUpts2', 'PGI', 'PFK', 'FBA', 'TPI', 'GAPD', 'PGK', 'PGM', 'ENO', 'PYK', 'EX_glc__D_e', 'EX_fru_e'],
        'Gluconeogenesis': ['FBP', 'PPS'],
        'PP Pathway': ['G6PDH2r', 'PGL', 'GND', 'RPE', 'RPI', 'TKT1', 'TKT2', 'TALA'],
        
        # TCA Cycle & Related
        'TCA Cycle': ['PDH', 'CS', 'ACONTa', 'ACONTb', 'ICDHyr', 'AKGDH', 'SUCOAS', 'SUCDi', 'FUM', 'MDH', 'FUMt2_2', 'MALt2_2',
            'SUCCt2_2', 'SUCCt3', 'AKGt2r', 'EX_akg_e', 'EX_succ_e', 'EX_fum_e', 'EX_mal__L_e'],
        'Glyoxylate Shunt': ['ICL', 'MALS'],
        'Anaplerotic': ['PPC', 'PPCK', 'ME1', 'ME2'],
        
        # Respiration & Energy
        'OxPhos': ['CYTBD', 'ATPS4r', 'NADH16', 'THD2', 'NADTRHD'],
        'ATP Maintenance': ['ATPM', 'ADK1'],
        'Anaerobic Respiration': ['FRD7'],
        
        # Nitrogen Metabolism
        'Nitrogen Uptake': ['GLNabc', 'GLUt2r', 'NH4t', 'EX_nh4_e', 'EX_gln__L_e', 'EX_glu__L_e'],
        'Glu/Gln Synthesis': ['GLNS', 'GLUDy', 'GLUN', 'GLUSy'],
        
        # Transport & EX (remaining)
        'Transport & EX': [
            'CO2t', 'H2Ot', 'O2t', 'PIt2r', 'PYRt2',
            'EX_co2_e', 'EX_h_e', 'EX_h2o_e', 'EX_o2_e', 'EX_pi_e', 'EX_pyr_e'
        ],

        # Biomass
        'Biomass': ['Biomass_Ecoli_core'],
    }
    
    # Create reverse mapping from flux to group
    flux_to_group = {}
    for group, fluxes in pathway_groups.items():
        for flux in fluxes:
            flux_to_group[flux] = group
    
    # Assign group IDs to each flux
    group_ids = []
    for flux in clean_fluxes:
        group_name = 'Other'  # Default group
        for key in flux_to_group:
            if key in flux:
                group_name = flux_to_group[key]
                break
        group_ids.append(group_name)
    
    group_color_map = {}
    for group in set(group_ids):
        if group in group_colors:
            group_color_map[group] = group_colors[group]
        else:
            group_color_map[group] = '#999999'
    
    # Get embeddings
    model.eval()
    all_embeddings = []
    flux_indices = []
    
    with torch.no_grad():
        batch_size = 128
        for i in range(0, n_samples, batch_size):
            batch = X_test[i:i+batch_size, :20].unsqueeze(-1).to(device)
            batch_context = torch.zeros(batch.size(0), model.vocab_size, 1, device=device)
            batch_context[:, :20] = batch
            
            embeddings = model(batch_context, return_embedding=True)
            output_embeddings = embeddings[:, 20:20+n_outputs, :]
            
            all_embeddings.append(output_embeddings.cpu())
            flux_indices.extend([np.arange(n_outputs)] * len(batch))
    
    # Combine all samples
    all_embeddings = torch.cat(all_embeddings, dim=0)
    flux_indices = np.concatenate(flux_indices)
    
    # Flatten for t-SNE
    embeddings_flat = all_embeddings.reshape(-1, all_embeddings.shape[-1]).numpy()
    
    # Run t-SNE
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings_flat)
    
    # Create color array for each point
    point_colors = [group_color_map[group_ids[flux_idx]] 
                   for flux_idx in flux_indices]
    
    plt.figure(figsize=(16, 12))
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
                c=point_colors, alpha=0.2, s=10)
    
    ordered_groups = list(group_colors.keys())
    legend_patches = []
    for group in ordered_groups:
        if group in set(group_ids):
            legend_patches.append(
                mpatches.Patch(color=group_colors[group], label=group)
            )

    print("Legend order:", [g for g in ordered_groups if g in set(group_ids)])

    plt.legend(handles=legend_patches, title="Metabolic Pathways",
               bbox_to_anchor=(1.05, 1), loc='upper left')
    for flux_idx in range(n_outputs):
        mask = flux_indices == flux_idx
        if np.any(mask):
            center = np.median(embeddings_2d[mask], axis=0)
            flux_name = clean_fluxes[flux_idx]
            flux = flux_name + '_flux'
            if flux in output_cols:
                plt.annotate(flux_name, center, xytext=(5, 5), 
                            textcoords='offset points', fontsize=9,
                            bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7))
    
    plt.title(f'Flux Embeddings Grouped by Metabolic Pathway\n({n_samples} input conditions)', fontsize=16)
    plt.xlabel('t-SNE 1', fontsize=12)
    plt.ylabel('t-SNE 2', fontsize=12)
    plt.grid(alpha=0.1)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
        print(f"Grouped t-SNE plot saved to {save_path}")
    else:
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    #set_seed()
    
    d_model = 128
    n_heads = 8
    n_layers = 3
    d_ff = 640
    batch_size = 128
    num_epochs = 200
    learning_rate = 1e-4
    dropout = 0.02
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    X, y, input_cols, output_cols = load_data(DATA_PATH)
    X_train, X_test, y_train, y_test = prepare_tensors(X, y, device=device)
    train_loader, test_loader = create_dataloaders(X_train, y_train, X_test, y_test, batch_size)

    train_loss, test_loss, model, optimizer = train_model(d_model, n_heads, n_layers, d_ff, num_epochs, learning_rate, dropout)

    today = date.today().isoformat()
    model_name = f"ecoli_core_d{d_model}_h{n_heads}_l{n_layers}_ff{d_ff}"
    pic_dir = f"./pics/{today}/{model_name}"
    os.makedirs(pic_dir, exist_ok=True)

    model_save_dir = f"./models/{model_name}"
    model_save_path = f"{model_save_dir}/{model_name}.pth"
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    torch.save(model.state_dict(), model_save_path)
    print(f"\nModel saved to {model_save_path}")

    checkpoint = {
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_losses': train_loss,
        'test_losses': test_loss,
        'config': {
            'd_model': d_model,
            'n_heads': n_heads,
            'n_layers': n_layers,
            'd_ff': d_ff,
            'dropout': dropout,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'num_epochs': num_epochs,
            'vocab_size': 115,
        },
        'rng_state': {
            'torch': torch.get_rng_state(),
            'numpy': np.random.get_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        },
        'data_info': {
            'dataset': DATA_PATH,
            'input_cols': input_cols,
            'output_cols': output_cols,
            'n_train': len(X_train),
            'n_test': len(X_test)
        }
    }
    
    checkpoint_path = f"{model_save_dir}/{model_name}_checkpoint.pth"
    torch.save(checkpoint, checkpoint_path)
    print(f"Full checkpoint saved to {checkpoint_path}")

    plot_loss_curves(
        train_loss, test_loss, 
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_ff,
        save_path=f"{pic_dir}/training_curve.png"
    )
    plot_prediction_sample(X_test, y_test, model, save_path=f"{pic_dir}/prediction_sample.png")

'''
    metrics = []

    model_cpu = model.to('cpu')  # move model to CPU 
    model_cpu.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to('cpu')
            preds = model_cpu(batch_X)
            all_preds.append(preds.cpu())
            all_targets.append(batch_y.cpu())

    # Concatenate predictions
    y_pred_tensor = torch.cat(all_preds, dim=0)
    y_pred = y_pred_tensor.numpy()[:, 20:]
    y_true = torch.cat(all_targets, dim=0).numpy()[:, 20:]

    for i, label in enumerate(output_cols):
        r2 = r2_score(y_true[:, i], y_pred[:, i])
        mae = mean_absolute_error(y_true[:, i], y_pred[:, i])
        print(f"{label}: R² = {r2:.3f}, MAE = {mae:.3f}")
        metrics.append({
            'flux': label,
            'r2': r2,
            'mae': mae,
        })
        plot_diagnostics_2x2(
            y_true[:, i],
            y_pred[:, i],
            label=label,
            save_path=f"{pic_dir}/{label}.png"
        )
    print(f'Diagnostic plots for all fluxes saved to {pic_dir}')
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(f"{pic_dir}/flux_metrics.csv", index=False)
    print(f"\nSaved R² and MAE metrics to {pic_dir}/flux_metrics.csv")
    

    # Generate all three types of embedding visualizations
    plot_pre_attention_embeddings_tsne(
        model_cpu, 
        output_cols,
        save_path=f"{pic_dir}/tsne_pre_attention_embeddings.png"
    )

    plot_post_attention_zero_context_tsne(
        model_cpu,
        output_cols,
        save_path=f"{pic_dir}/tsne_post_attention_zero_c.png"
    )
    
    plot_post_attention_real_context(
        model_cpu, 
        output_cols, 
        X_test, 
        method='tsne', 
        n_components=2, 
        n_samples=1000, 
        perplexity=30.0,
        save_path=f"{pic_dir}/tsne_post_attention_embeddings.png"
    )

    plot_post_attention_grouped_tsne(
        model, 
        output_cols, 
        X_test, 
        n_samples=1000, 
        perplexity=30.0, 
        save_path=f"{pic_dir}/tsne_post_attention_grouped.png"
    )
    
    plot_post_attention_real_context(
        model, 
        output_cols, 
        X_test, 
        method='umap',
        n_neighbors=50,
        min_dist=0.5,
        save_path=f"{pic_dir}/umap_post_attention_real_c.png"
    )
    '''
