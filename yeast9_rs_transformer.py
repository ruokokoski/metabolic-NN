import argparse
import gc
import os
import random
import time
from collections import Counter
from datetime import date

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from yeast9_reactions import inputs, outputs

sample_counter = Counter()

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

DATA_PATH = "./data/2025-11-07_yeast9_data_246923_samples.csv"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Train the Yeast9 FluxTransformer on the full output vocabulary "
            "or correlation-based output subsets."
        )
    )
    parser.add_argument(
        "--sampling-strategy",
        choices=("full", "correlation"),
        default="full",
        help="Output-token training strategy (default: full).",
    )
    parser.add_argument(
        "--output-sample-ratio",
        type=float,
        default=0.5,
        help="Fraction of output tokens selected in correlation mode (default: 0.5).",
    )
    parser.add_argument(
        "--cov-prob",
        type=float,
        default=0.7,
        help="Probability of choosing a correlated group instead of a uniform subset (default: 0.7).",
    )
    return parser.parse_args(argv)

class AttentionBlock(nn.Module):
    """Custom multi-head attention block for metabolic modeling"""
    def __init__(self, d_model=128, n_heads=8, dropout=0.2):
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
    def __init__(self, d_model, d_ff, dropout=0.2):
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
    def __init__(self, d_model=128, n_heads=8, d_ff=1024, dropout=0.2):
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
        vocab_size=2742,
        d_model=128,
        n_heads=8,
        n_layers=3,
        d_ff=1024,
        dropout=0.2,
        input_length=30
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
            selected_indices: indices of tokens used (always include inputs 0..29)
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

        for layer in self.layers:
            x, c_subset = layer(x, c_subset)

        if return_embedding:
            return x, selected_indices  # embeddings + indices

        return c_subset, selected_indices

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
    
    df = pd.read_csv(filepath)

    # Fill missing inputs with 0 (i.e., not uptaken)
    df[inputs] = df[inputs].fillna(0)

    print(f"\nLoaded data with {len(df)} samples from {filepath}")
    print(f"Number of input features: {len(inputs)}")
    print(f"Number of output targets: {len(outputs)}")

    X = df[inputs].to_numpy(dtype=np.float32)
    y = df[[f"{col}_flux" for col in outputs]].to_numpy(dtype=np.float32)

    X_combined = np.hstack([X, np.zeros_like(y)])
    y_combined = np.hstack([np.zeros_like(X), y])

    return X_combined, y_combined, inputs, outputs

def prepare_tensors(X, y, test_size=0.2):
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

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

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

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

    return train_loader, test_loader

def precompute_flux_correlation(data_path, output_cols, device="cpu"):
    """Compute the signed Pearson correlation matrix for the output fluxes."""
    df = pd.read_csv(data_path)
    flux_data = df[[f"{column}_flux" for column in output_cols]].to_numpy(
        dtype=np.float32
    )
    flux_centered = flux_data - flux_data.mean(axis=0)

    covariance = np.cov(flux_centered, rowvar=False)
    standard_deviation = np.sqrt(np.diag(covariance))
    denominator = standard_deviation[:, None] * standard_deviation[None, :]
    denominator[denominator == 0] = 1.0
    correlation = np.nan_to_num(covariance / denominator)
    corr_tensor = torch.tensor(correlation, dtype=torch.float32, device=device)

    print(
        f"\nCorrelation matrix computed: shape {corr_tensor.shape}, "
        f"min {corr_tensor.min():.3f}, max {corr_tensor.max():.3f}"
    )
    return corr_tensor


def sample_output_subset(
    total_outputs,
    output_start_idx,
    corr_tensor,
    output_sample_ratio=0.5,
    cov_prob=0.7,
):
    """Select a positively correlated output group or a uniform output subset."""
    device = corr_tensor.device
    n_sampled = max(1, int(total_outputs * output_sample_ratio))

    if np.random.rand() < cov_prob:
        primary_idx = torch.randint(0, total_outputs, (1,), device=device)
        correlations = corr_tensor[primary_idx.item()].clone()
        correlations[primary_idx.item()] = -1
        top_correlated = torch.topk(correlations, n_sampled - 1).indices
        selected = torch.cat([primary_idx, top_correlated])
    else:
        selected = torch.randperm(total_outputs, device=device)[:n_sampled]

    return selected + output_start_idx


def compute_output_sampling_weights(
    data_path=DATA_PATH,
    inputs_list=inputs,
    outputs_list=outputs,
    boost_dict=None,   # e.g. {"r_2111_flux": 6.5}
    min_value=1e-8,
    temperature=1.0
):
    """
    Returns:
        weights: numpy array of length len(outputs_list) with normalized probabilities
        stats: DataFrame with abs_mean, variance, log_abs_mean for all flux columns (index includes '_flux')
    """
    df_all = pd.read_csv(data_path)
    df = df_all.drop(columns=inputs_list, errors='ignore')

    # ensure flux columns are present with _flux suffix
    flux_cols = [f"{o}_flux" for o in outputs_list]

    abs_mean = df.abs().mean()
    variance = df.var()
    log_abs_mean = np.log1p(abs_mean)   # log(1 + abs_mean)

    stats = pd.DataFrame({
        "abs_mean": abs_mean,
        "variance": variance,
        "log_abs_mean": log_abs_mean
    })

    # Build weight vector for outputs in the same order as outputs_list
    w_list = []
    for col in flux_cols:
        if col in stats.index:
            w_list.append(stats.loc[col, "log_abs_mean"])
        else:
            w_list.append(min_value)

    weights = np.array(w_list, dtype=np.float64)

    # boost given reactions
    if boost_dict:
        for rxn_name, factor in boost_dict.items():
            if not rxn_name.endswith("_flux"):
                rxn_key = f"{rxn_name}_flux"
            else:
                rxn_key = rxn_name
            if rxn_key in flux_cols:
                idx = flux_cols.index(rxn_key)
                weights[idx] = weights[idx] * float(factor)

    # replace NaN, non-finite or zero with small weight
    weights = np.nan_to_num(weights, nan=min_value, posinf=min_value, neginf=min_value)
    weights = np.where(weights == 0, min_value, weights)

    # Softmax with temperature
    exp_weights = np.exp(weights / temperature)
    weights = exp_weights / np.sum(exp_weights)

    return weights, stats

def train_model(
        d_model=128, 
        n_heads=8, 
        n_layers=3, 
        d_ff=1024, 
        num_epochs=100, 
        learning_rate=0.001, 
        dropout=0.02, 
        model_name="yeast9",
        sampling_strategy="full",
        output_sample_ratio=0.5,
        cov_prob=0.7,
        corr_tensor=None,
        validation_weights=None,
    ):
    start_time = time.time()

    model_save_dir = f"./models/{model_name}"
    os.makedirs(model_save_dir, exist_ok=True)
    checkpoint_path = f"{model_save_dir}/{model_name}_checkpoint.pth"

    model = FluxTransformer(
        vocab_size=len(inputs) + len(outputs),
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_ff,
        dropout=dropout,
        input_length=len(inputs)
    ).to(device)
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.98),
        eps=1e-6,
        weight_decay=1e-4
    )
    criterion = nn.HuberLoss()

    train_losses, test_losses = [], []
    start_epoch, best_test_loss, best_epoch = 0, float("inf"), -1

    total_outputs = len(outputs)
    output_start_idx = len(inputs)

    if sampling_strategy not in {"full", "correlation"}:
        raise ValueError("sampling_strategy must be 'full' or 'correlation'")
    if not 0 < output_sample_ratio <= 1:
        raise ValueError("output_sample_ratio must be in (0, 1]")
    if not 0 <= cov_prob <= 1:
        raise ValueError("cov_prob must be in [0, 1]")

    fixed_eval_global = None
    if sampling_strategy == "correlation":
        if corr_tensor is None:
            raise ValueError("corr_tensor is required for correlation sampling")
        if validation_weights is None:
            raise ValueError("validation_weights are required for correlation sampling")

        n_eval_outputs = min(256, total_outputs)
        fixed_eval_relative = np.argsort(validation_weights)[-n_eval_outputs:]
        fixed_eval_global = torch.tensor(
            fixed_eval_relative + output_start_idx,
            device=device,
        )

    if os.path.exists(checkpoint_path):
        print(f"\nResuming training from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        train_losses = checkpoint.get("train_losses", [])
        test_losses = checkpoint.get("test_losses", [])
        completed_epochs = checkpoint.get("epoch", 0)
        start_epoch = completed_epochs
        best_test_loss = min(test_losses) if test_losses else float("inf")
        best_epoch = int(np.argmin(test_losses) + 1) if test_losses else -1

        if start_epoch >= num_epochs:
            print(f"Checkpoint indicates {start_epoch} completed epochs which is >= requested num_epochs ({num_epochs}).")
            print("No training to do. If you want to continue training, pass a larger num_epochs.")
            return train_losses, test_losses, model, optimizer
    else:
        print("\nNo checkpoint found. Starting fresh training.")

    for epoch in range(start_epoch, num_epochs):
        model.train()
        epoch_train_loss = 0.0
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)

            if sampling_strategy == "full":
                sampled_indices = None
            else:
                sampled_indices = sample_output_subset(
                    total_outputs=total_outputs,
                    output_start_idx=output_start_idx,
                    corr_tensor=corr_tensor,
                    output_sample_ratio=output_sample_ratio,
                    cov_prob=cov_prob,
                )

            predictions, selected_indices = model(batch_X, output_subset=sampled_indices)

            sample_counter.update(selected_indices.cpu().numpy().tolist())

            out_mask = selected_indices >= output_start_idx
            pred_out = predictions[:, out_mask, :]
            target_full = batch_y[:, selected_indices, :]
            tgt_out = target_full[:, out_mask, :]

            loss = criterion(pred_out, tgt_out)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_train_loss += loss.item() * batch_X.size(0)

            # Explicitly free tensors
            del predictions, loss, pred_out, tgt_out, target_full, batch_X, batch_y
        
        epoch_train_loss /= len(train_loader.dataset)
        train_losses.append(epoch_train_loss)

        # Evaluation
        model.eval()
        epoch_test_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X = batch_X.to(device, non_blocking=True)
                batch_y = batch_y.to(device, non_blocking=True)

                predictions, selected_indices = model(
                    batch_X,
                    output_subset=fixed_eval_global,
                )
        
                out_mask = selected_indices >= output_start_idx
                pred_out = predictions[:, out_mask, :]
                
                # Get corresponding targets
                target_full = batch_y[:, selected_indices, :]
                tgt_out = target_full[:, out_mask, :]

                loss = criterion(pred_out, tgt_out)
                epoch_test_loss += loss.item() * batch_X.size(0)

                # Explicitly free tensors
                del predictions, loss, pred_out, tgt_out, target_full, batch_X, batch_y
        
        epoch_test_loss /= len(test_loader.dataset)
        test_losses.append(epoch_test_loss)

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

def calculate_metrics(model, X_test, y_test, inputs, batch_size=64, device=None):
    """
    Compute overall metrics (R^2 and MAE) on the output columns in mini-batches.
    """
    n_inputs = len(inputs) if not isinstance(inputs, int) else inputs
    device = device or next(model.parameters()).device

    model.eval()
    preds_list, trues_list = [], []

    with torch.no_grad():
        for i in range(0, X_test.size(0), batch_size):
            xb = X_test[i:i+batch_size].to(device)
            yb = y_test[i:i+batch_size].to(device)

            # forward pass
            pb, _ = model(xb.unsqueeze(-1))  # [B, V, 1]

            if yb.dim() == 2:
                yb = yb.unsqueeze(-1)

            preds_list.append(pb[:, n_inputs:, 0].cpu())
            trues_list.append(yb[:, n_inputs:, 0].cpu())

    pred_outputs = torch.cat(preds_list).numpy()
    true_outputs = torch.cat(trues_list).numpy()

    # compute metrics
    r2 = r2_score(true_outputs.ravel(), pred_outputs.ravel())
    mae = mean_absolute_error(true_outputs.ravel(), pred_outputs.ravel())

    print(f"Overall R²: {r2:.4f}")
    print(f"Overall MAE: {mae:.4f}")

    return {"r2": r2, "mae": mae}

if __name__ == "__main__":
    args = parse_args()
    set_seed()
    
    d_model = 128
    n_heads = 8
    n_layers = 3
    d_ff = 1024
    batch_size = 4
    num_epochs = 60
    learning_rate = 1e-4
    dropout = 0.02
    sampling_strategy = args.sampling_strategy
    output_sample_ratio = args.output_sample_ratio
    cov_prob = args.cov_prob
    temperature = 5.0
    model_name = f"yeast9_d{d_model}_h{n_heads}_l{n_layers}_ff{d_ff}_{sampling_strategy}"
    if sampling_strategy == "correlation":
        ratio_tag = str(output_sample_ratio).replace(".", "_")
        cov_prob_tag = str(cov_prob).replace(".", "_")
        model_name += f"_r{ratio_tag}_p{cov_prob_tag}"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    X, y, input_cols, output_cols = load_data(DATA_PATH)
    X_train, X_test, y_train, y_test = prepare_tensors(X, y)
    train_loader, test_loader = create_dataloaders(X_train, y_train, X_test, y_test, batch_size)

    corr_tensor = None
    validation_weights = None
    if sampling_strategy == "correlation":
        corr_tensor = precompute_flux_correlation(DATA_PATH, outputs, device=device)
        t0 = time.time()
        validation_weights, _ = compute_output_sampling_weights(
            data_path=DATA_PATH,
            inputs_list=inputs,
            outputs_list=outputs,
            boost_dict=None,
            temperature=temperature,
        )
        print(
            f"Prepared validation weights for {len(validation_weights)} outputs. "
            f"Sum={validation_weights.sum():.6f}"
        )
        print(f"Weight computation took {time.time() - t0:.0f} seconds")

    print(f"Sampling strategy: {sampling_strategy}")
    if sampling_strategy == "correlation":
        print(f"Output sample ratio: {output_sample_ratio}")
        print(f"Correlation probability: {cov_prob}")

    train_loss, test_loss, model, optimizer = train_model(
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_ff,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        dropout=dropout,
        model_name=model_name,
        sampling_strategy=sampling_strategy,
        output_sample_ratio=output_sample_ratio,
        cov_prob=cov_prob,
        corr_tensor=corr_tensor,
        validation_weights=validation_weights,
    )

    today = date.today().isoformat()

    model_save_dir = f"./models/{model_name}"
    model_save_path = f"{model_save_dir}/{model_name}.pth"
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    torch.save(model.state_dict(), model_save_path)

    print(f"\nModel saved to {model_save_path}")

    checkpoint = {
        'epoch': len(train_loss),
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
            'vocab_size': len(inputs) + len(outputs),
            'n_inputs': len(inputs),
            'sampling_strategy': sampling_strategy,
            'output_sample_ratio': 1.0 if sampling_strategy == 'full' else output_sample_ratio,
            'cov_prob': cov_prob if sampling_strategy == 'correlation' else None,
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
    if sampling_strategy == "correlation":
        print(f"Validation-weight temperature = {temperature}")
    print(f"Full checkpoint saved to {checkpoint_path}")

    hist_path = os.path.join(model_save_dir, f"{model_name}_selected_indices.txt")

    with open(hist_path, "w") as f:
        for idx, count in sorted(sample_counter.items()):
            f.write(f"{idx}: {count}\n")

    print(f"Selected indices histogram saved to {hist_path}")

    #metrics = calculate_metrics(model, X_test, y_test, inputs, batch_size=batch_size, device=device)

