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

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DATA_PATH = "./data/2025-07-15_full_training_data_98066_samples.csv"

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
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
    """Custom multi-head attention block for metabolic modeling"""
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
            batch_first=True
        )

        self.head_scores = nn.Parameter(torch.zeros(n_heads))

    def forward(self, x, c):
        # x: (batch, seq_len, d_model)
        # c: (batch, seq_len, 1)
        x_norm = self.layer_norm(x) # pre-norm
        attn_out, attn_weights = self.mha(x_norm, x_norm, x_norm, need_weights=True, average_attn_weights=False)

        x_out = attn_out + x

        # Per-head diffusion of c:
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
    def __init__(self, d_model=128, n_heads=8, d_ff=1024, dropout=0.05):
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
        vocab_size=95,
        d_model=128,
        n_heads=8,
        n_layers=3,
        d_ff=1024,
        dropout=0.05,
        input_length=20
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.input_length = input_length

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

    def forward(self, c, output_subset=None, return_embedding=False):
        """
        Args:
            c: (batch, 1) or (batch, seq, 1) context tensor
            output_subset: 1D tensor of indices (subset of outputs) or None
            return_embedding: if True, return embeddings instead of c

        Returns:
            c: updated context
            selected_indices: indices of tokens used
        """
        batch_size = c.size(0)

        # Always include input indices
        input_indices = torch.arange(self.input_length, device=c.device)

        if output_subset is None:
            selected_indices = torch.arange(self.vocab_size, device=c.device)
        else:
            # Concatenate inputs + sampled outputs
            selected_indices = torch.cat([input_indices, output_subset.to(c.device)])
            selected_indices = torch.unique(selected_indices, sorted=True)

        # Expand indices for batch
        y = selected_indices.unsqueeze(0).expand(batch_size, -1)  # (B, seq_subset)
        x = self.input_embedding(y)  # (B, seq_subset, d_model)

        # Slice c to selected indices
        c_subset = c[:, selected_indices, :]
        c_subset_all_layers = torch.zeros(batch_size, c_subset.size()[1], len(self.layers), device=c.device)

        for e, layer in enumerate(self.layers):
            x, c_subset = layer(x, c_subset)
            c_subset_all_layers[:, :, e] = c_subset.squeeze(-1)

        if return_embedding:
            return x, selected_indices  # embeddings + indices

        return c_subset_all_layers, selected_indices

def load_data(filepath):
    """
    Vocabulary = outputs only (V = len(outputs)).
    Inputs are injected as scalars into the FIRST K token slots by reordering outputs so that
    the K exchange flux tokens corresponding to 'inputs' come first.

    Returns:
      X_tok: (N, V)  inputs injected into first K positions, rest 0
      y_tok: (N, V)  first K positions forced to 0, remaining are true outputs
      inputs, outputs_ordered: lists
      input_length, out_indices, perm (optional)
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
    df[inputs] = df[inputs].fillna(0)

    # Map each input token name -> corresponding output-token name
    input_flux_tokens = [f"{name}_flux" for name in inputs]

    # Validate mapping exists in outputs
    missing = [t for t in input_flux_tokens if t not in outputs]
    if missing:
        raise ValueError(
            "These mapped input flux tokens are missing from outputs:\n"
            + "\n".join(missing)
        )

    # Reorder outputs: (input-mapped exchange flux tokens) first, then the rest
    outputs_ordered = input_flux_tokens + [o for o in outputs if o not in set(input_flux_tokens)]

    # Build permutation from original outputs -> ordered outputs
    perm = [outputs.index(o) for o in outputs_ordered]

    # Read matrices
    X_in = df[inputs].to_numpy(dtype=np.float32)            # (N, K)
    y_out = df[outputs].to_numpy(dtype=np.float32)          # (N, V) original order
    y_out = y_out[:, perm]                                  # (N, V) reordered to outputs_ordered

    K = len(inputs)
    V = len(outputs_ordered)

    # Model input c: inject inputs into first K token slots; rest 0
    X_tok = np.zeros((len(df), V), dtype=np.float32)
    X_tok[:, :K] = X_in

    # Training target: force input slots to 0, train the rest to match y
    y_tok = y_out.copy()
    y_tok[:, :K] = 0.0

    out_indices = list(range(K, V))

    print(f"\nLoaded data with {len(df)} samples from {filepath}")
    print(f"Inputs (K): {K} injected into token slots [0..{K-1}]")
    print(f"Vocab/outputs (V): {V} (outputs_ordered)")
    return X_tok, y_tok, inputs, outputs_ordered, K, out_indices


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

def _empty_cache(device):
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()

def train_model(
    train_loader,
    test_loader,
    device,
    input_length,      # K
    out_indices,       # indices K..V-1
    vocab_size,        # V == len(outputs_ordered)
    d_model,
    n_heads,
    n_layers,
    d_ff,
    num_epochs,
    learning_rate,
    dropout,
    output_sample_ratio=1.0
):
    start_time = time.time()

    model = FluxTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_ff,
        dropout=dropout,
        input_length=input_length  # <-- K
    ).to(device)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.98),
        eps=1e-9,
        weight_decay=1e-4
    )
    criterion = nn.HuberLoss()

    best_test_loss = float("inf")
    best_epoch = -1
    train_losses, test_losses = [], []

    total_outputs = len(out_indices)

    for epoch in range(num_epochs):
        model.train()
        epoch_train_loss = 0.0

        for batch_X, batch_y in train_loader:
            optimizer.zero_grad(set_to_none=True)

            if output_sample_ratio >= 1.0:
                sampled_indices = None
            else:
                n_sampled = max(1, int(total_outputs * output_sample_ratio))
                sampled_indices = torch.tensor(
                    random.sample(out_indices, n_sampled),
                    device=device,
                    dtype=torch.long
                )

            preds_all_layers, selected_indices = model(batch_X, output_subset=sampled_indices)

            # last-layer scalar prediction: (B, S, 1)
            pred_last = preds_all_layers[:, :, -1].unsqueeze(2)
            target = batch_y[:, selected_indices, :]  # (B, S, 1)

            loss = criterion(pred_last, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_train_loss += loss.item() * batch_X.size(0)

            del preds_all_layers, pred_last, target, loss
            _empty_cache(device)

        epoch_train_loss /= len(train_loader.dataset)
        train_losses.append(epoch_train_loss)

        model.eval()
        epoch_test_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                if output_sample_ratio >= 1.0:
                    sampled_indices = None
                else:
                    n_sampled = max(1, int(total_outputs * output_sample_ratio))
                    sampled_indices = torch.tensor(
                        random.sample(out_indices, n_sampled),
                        device=device,
                        dtype=torch.long
                    )

                preds_all_layers, selected_indices = model(batch_X, output_subset=sampled_indices)
                pred_last = preds_all_layers[:, :, -1].unsqueeze(2)
                target = batch_y[:, selected_indices, :]

                loss = criterion(pred_last, target)
                epoch_test_loss += loss.item() * batch_X.size(0)

                del preds_all_layers, pred_last, target, loss
                _empty_cache(device)

        epoch_test_loss /= len(test_loader.dataset)
        test_losses.append(epoch_test_loss)

        if (epoch + 1) % 2 == 0:
            print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {epoch_train_loss:.6f} | Test Loss: {epoch_test_loss:.6f}")

        if epoch_test_loss < best_test_loss:
            best_test_loss = epoch_test_loss
            best_epoch = epoch + 1

        _empty_cache(device)
        gc.collect()

    elapsed = time.time() - start_time
    mins, secs = divmod(elapsed, 60)
    print("Training Completed.")
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
    output_sample_ratio = 1.0
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    X, y, inputs, outputs_ordered, K, out_indices = load_data(DATA_PATH)
    X_train, X_test, y_train, y_test = prepare_tensors(X, y, test_size=0.2, device=device)
    train_loader, test_loader = create_dataloaders(X_train, y_train, X_test, y_test, batch_size)

    train_loss, test_loss, model, optimizer = train_model(
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        input_length=K,
        out_indices=out_indices,
        vocab_size=len(outputs_ordered),
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_ff,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        dropout=dropout,
        output_sample_ratio=output_sample_ratio
    )

    today = date.today().isoformat()
    model_name = f"ecoli_core_merged_d{d_model}_h{n_heads}_l{n_layers}_ff{d_ff}"
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
            'vocab_size': len(outputs_ordered),
            'input_length': K
        },
        'rng_state': {
            'torch': torch.get_rng_state(),
            'numpy': np.random.get_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        },
        'data_info': {
            'dataset': DATA_PATH,
            'input_cols': inputs,
            'output_cols': outputs_ordered,
            'n_train': len(X_train),
            'n_test': len(X_test)
        }
    }
    
    checkpoint_path = f"{model_save_dir}/{model_name}_checkpoint.pth"
    torch.save(checkpoint, checkpoint_path)
    print(f"Full checkpoint saved to {checkpoint_path}")

    '''
    plot_loss_curves(
        train_loss, test_loss, 
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_ff,
        save_path=f"{pic_dir}/training_curve.png"
    )
    '''
