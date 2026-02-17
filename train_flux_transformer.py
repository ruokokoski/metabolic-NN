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

#DATA_PATH = "./data/2025-07-28_full_training_data_98066_samples.csv"
DATA_PATH = "./data/iML1515_exp_212000_samples.csv"
#MODEL_NAME = "ecoli_core"
MODEL_NAME = "iML1515_exp"

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)

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
    elif torch.backends.mps.is_available():
        allocated = torch.mps.current_allocated_memory() / 1024**2
        reserved = torch.mps.driver_allocated_memory() / 1024**2
        print(f'MPS Allocated memory: {allocated:.2f} MB')
        print(f'MPS Driver allocated memory: {reserved:.2f} MB')
    else:
        print("CUDA or MPS is not available.")

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
        vocab_size: int,
        input_token_indices,          # list[int] or 1D tensor[int64]
        d_model=128,
        n_heads=8,
        n_layers=3,
        d_ff=1024,
        dropout=0.05,
    ):
        super().__init__()
        if vocab_size is None:
            raise ValueError("vocab_size must be provided explicitly.")
        self.vocab_size = int(vocab_size)
        self.d_model = d_model

        idx = torch.as_tensor(input_token_indices, dtype=torch.long)
        if idx.ndim != 1:
            raise ValueError("input_token_indices must be 1D.")
        if (idx < 0).any() or (idx >= self.vocab_size).any():
            raise ValueError("input_token_indices contains out-of-range indices.")
        # Register as buffer so it moves with .to(device)
        self.register_buffer("input_token_indices", idx, persistent=True)

        self.input_embedding = nn.Embedding(self.vocab_size, d_model)

        self.layers = nn.ModuleList([
            FluxTransformerLayer(d_model=d_model, n_heads=n_heads, d_ff=d_ff, dropout=dropout)
            for _ in range(n_layers)
        ])

    def forward(self, c, output_subset=None, return_embedding=False):
        """
        c: (batch, vocab_size, 1)
        output_subset: 1D tensor of token indices to train on (typically excludes injected tokens)
        """
        batch_size = c.size(0)

        always = self.input_token_indices  # (n_injected,)

        if output_subset is None:
            selected_indices = torch.arange(self.vocab_size, device=c.device)
        else:
            output_subset = output_subset.to(c.device).long()
            selected_indices = torch.unique(torch.cat([always, output_subset]), sorted=True)

        y = selected_indices.unsqueeze(0).expand(batch_size, -1)      # (B, S)
        x = self.input_embedding(y)                                    # (B, S, d_model)

        c_subset = c[:, selected_indices, :]                           # (B, S, 1)
        c_subset_all_layers = torch.zeros(batch_size, c_subset.size(1), len(self.layers), device=c.device)

        for e, layer in enumerate(self.layers):
            x, c_subset = layer(x, c_subset)
            c_subset_all_layers[:, :, e] = c_subset.squeeze(-1)

        if return_embedding:
            return x, selected_indices

        return c_subset_all_layers, selected_indices


def load_data(filepath):
    """
    Vocabulary = outputs only (token list == `outputs` inferred from CSV header order).

      - Inputs are all leading columns before the first `*_flux` column.
      - Outputs are all columns from the first `*_flux` column onward.
      - X_tok: zeros everywhere, except at indices of the exchange flux tokens
               (e.g. 'EX_glc__D_e_flux') where we write the medium constraint value
               from input column 'EX_glc__D_e'.
      - y_tok: realized fluxes for all output tokens.

    Returns:
      X_tok: (n_samples, n_tokens)
      y_tok: (n_samples, n_tokens)
      inputs: list[str]          (medium constraint columns)
      outputs: list[str]         (flux token columns, defines token order)
      input_token_indices: list[int]  indices in `outputs` corresponding to each input's *_flux token
      out_indices: list[int]          all other token indices (non-injected tokens)
    """
    df = pd.read_csv(filepath)
    columns = list(df.columns)

    first_flux_idx = next((i for i, col in enumerate(columns) if col.endswith("_flux")), None)
    if first_flux_idx is None:
        raise ValueError("No output columns found. Expected at least one column ending with '_flux'.")

    inputs = columns[:first_flux_idx]
    outputs = columns[first_flux_idx:]

    if not inputs:
        raise ValueError("No input columns found before the first '_flux' output column.")

    non_flux_outputs = [col for col in outputs if not col.endswith("_flux")]
    if non_flux_outputs:
        raise ValueError(
            "Found non-output columns after outputs started. "
            "Expected all columns from first output onward to end with '_flux':\n"
            + "\n".join(non_flux_outputs)
        )

    # Medium constraints: treat missing as "not provided" -> 0
    df[inputs] = df[inputs].fillna(0)

    input_flux_tokens = [f"{name}_flux" for name in inputs]
    missing = [t for t in input_flux_tokens if t not in outputs]
    if missing:
        raise ValueError(
            "These mapped input flux tokens are missing from outputs:\n" + "\n".join(missing)
        )

    n_samples = len(df)
    n_tokens = len(outputs)

    # Realized flux targets
    y_tok = df[outputs].to_numpy(dtype=np.float32)  # (n_samples, n_tokens)

    # Build X_tok by writing each medium value into its corresponding *_flux token index
    X_tok = np.zeros((n_samples, n_tokens), dtype=np.float32)

    X_in = df[inputs].to_numpy(dtype=np.float32)  # (n_samples, n_inputs)
    input_token_indices = []
    for j, tok in enumerate(input_flux_tokens):
        tok_idx = outputs.index(tok)
        input_token_indices.append(tok_idx)
        X_tok[:, tok_idx] = X_in[:, j]

    injected_set = set(input_token_indices)
    out_indices = [i for i in range(n_tokens) if i not in injected_set]

    print(f"\nLoaded data with {n_samples} samples from {filepath}")
    print(f"Extracted input columns count:  {len(inputs)}")
    print(f"Extracted output columns count: {len(outputs)}")
    print("Medium constraints are injected into these token indices (in outputs order):")
    print(input_token_indices)

    return X_tok, y_tok, inputs, outputs, input_token_indices, out_indices

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
    input_token_indices,   # list[int]
    out_indices,           # list[int] (non-injected tokens to sample from)
    vocab_size,            # int == len(outputs)
    d_model,
    n_heads,
    n_layers,
    d_ff,
    num_epochs,
    learning_rate,
    dropout,
    output_sample_ratio=1.0
):
    model = FluxTransformer(
        vocab_size=vocab_size,
        input_token_indices=input_token_indices,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_ff,
        dropout=dropout,
    ).to(device)

    start_time = time.time()

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

    train_meta = {
        "best_test_loss": float(best_test_loss),
        "best_epoch": int(best_epoch),
        "elapsed_sec": float(elapsed),
        "num_epochs": int(num_epochs),
        "output_sample_ratio": float(output_sample_ratio),
    }
    return train_losses, test_losses, model, optimizer, train_meta


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
    
    d_model = 256
    n_heads = 8
    n_layers = 3
    d_ff = 1024
    batch_size = 8
    num_epochs = 10
    learning_rate = 1e-4
    dropout = 0.02
    output_sample_ratio = 1.0
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    X, y, inputs, outputs, input_token_indices, out_indices = load_data(DATA_PATH)

    X_train, X_test, y_train, y_test = prepare_tensors(X, y, test_size=0.2, device=device)
    train_loader, test_loader = create_dataloaders(X_train, y_train, X_test, y_test, batch_size)

    train_loss, test_loss, model, optimizer, train_meta = train_model(
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        input_token_indices=input_token_indices,
        out_indices=out_indices,
        vocab_size=len(outputs),
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
    model_name = f"{MODEL_NAME}_d{d_model}_h{n_heads}_l{n_layers}_ff{d_ff}"
    pic_dir = f"./pics/{today}/{model_name}"
    os.makedirs(pic_dir, exist_ok=True)

    model_save_dir = f"./models/{model_name}"
    model_save_path = f"{model_save_dir}/{model_name}.pth"
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    torch.save(model.state_dict(), model_save_path)
    print(f"\nModel weights saved to {model_save_path}")

    checkpoint = {
        "epoch": num_epochs,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_losses": train_loss,
        "test_losses": test_loss,
        "config": {
            "d_model": d_model,
            "n_heads": n_heads,
            "n_layers": n_layers,
            "d_ff": d_ff,
            "dropout": dropout,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "num_epochs": num_epochs,
            "vocab_size": len(outputs),
            "input_token_indices": [int(i) for i in input_token_indices],
            "output_sample_ratio": output_sample_ratio,
        },
        "rng_state": {
            "torch": torch.get_rng_state(),
            "numpy": np.random.get_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "python": random.getstate(),
        },
        "data_info": {
            "dataset": DATA_PATH,
            "input_cols": inputs,
            "output_cols": outputs,
            "n_train": int(len(X_train)),
            "n_test": int(len(X_test)),
            "input_token_indices": [int(i) for i in input_token_indices],
            "out_indices": [int(i) for i in out_indices],
        },
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
