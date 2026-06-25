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
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

#DATA_PATH = "./data/2026-03-30_ecoli_core_training_data_9805381_samples.csv"
DATA_PATH = "./data/2026-06-25_ecoli_core_training_data_980621_samples.csv"
MODEL_NAME = "ecoli_core_1M"

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
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


def _count_rows(filepath):
    with open(filepath, "rb") as f:
        return sum(1 for _ in f) - 1


def cleanup_memmap_files(filepath, cache_dir="./tmp_memmap"):
    base = os.path.splitext(os.path.basename(filepath))[0]
    targets = [
        os.path.join(cache_dir, f"{base}_X_tok.float32.mmap"),
        os.path.join(cache_dir, f"{base}_y_tok.float32.mmap"),
    ]

    removed = []
    for path in targets:
        if os.path.exists(path):
            try:
                os.remove(path)
                removed.append(path)
            except OSError as err:
                print(f"Warning: could not remove {path}: {err}")

    if os.path.isdir(cache_dir) and not os.listdir(cache_dir):
        try:
            os.rmdir(cache_dir)
        except OSError:
            pass

    return removed


def load_data(filepath, cache_dir="./tmp_memmap", chunksize=200_000):
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
    columns = list(pd.read_csv(filepath, nrows=0).columns)

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

    input_flux_tokens = [f"{name}_flux" for name in inputs]
    missing = [t for t in input_flux_tokens if t not in outputs]
    if missing:
        raise ValueError(
            "These mapped input flux tokens are missing from outputs:\n" + "\n".join(missing)
        )

    n_samples = _count_rows(filepath)
    n_tokens = len(outputs)

    input_token_indices = [outputs.index(tok) for tok in input_flux_tokens]

    os.makedirs(cache_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(filepath))[0]
    x_path = os.path.join(cache_dir, f"{base}_X_tok.float32.mmap")
    y_path = os.path.join(cache_dir, f"{base}_y_tok.float32.mmap")

    X_tok = np.memmap(x_path, dtype=np.float32, mode="w+", shape=(n_samples, n_tokens))
    y_tok = np.memmap(y_path, dtype=np.float32, mode="w+", shape=(n_samples, n_tokens))
    X_tok[:] = 0.0

    usecols = inputs + outputs
    reader = pd.read_csv(
        filepath,
        usecols=usecols,
        chunksize=chunksize,
        dtype=np.float32,
    )

    row_start = 0
    for chunk_idx, chunk in enumerate(reader, start=1):
        if inputs:
            chunk[inputs] = chunk[inputs].fillna(0.0)

        n_chunk = len(chunk)
        row_end = row_start + n_chunk

        y_chunk = chunk[outputs].to_numpy(dtype=np.float32, copy=False)
        y_tok[row_start:row_end, :] = y_chunk

        if inputs:
            X_in = chunk[inputs].to_numpy(dtype=np.float32, copy=False)
            for j, tok_idx in enumerate(input_token_indices):
                X_tok[row_start:row_end, tok_idx] = X_in[:, j]

        row_start = row_end
        if chunk_idx % 10 == 0:
            print(f"Loaded {row_end:,}/{n_samples:,} rows...")

    X_tok.flush()
    y_tok.flush()

    injected_set = set(input_token_indices)
    out_indices = [i for i in range(n_tokens) if i not in injected_set]

    print(f"\nLoaded data with {n_samples} samples from {filepath}")
    print(f"Extracted input columns count:  {len(inputs)}")
    print(f"Extracted output columns count: {len(outputs)}")
    print("Medium constraints are injected into these token indices (in outputs order):")
    print(input_token_indices)

    return X_tok, y_tok, inputs, outputs, input_token_indices, out_indices


def prepare_split_indices(n_samples, test_size=0.2, random_state=42):
    indices = np.arange(n_samples, dtype=np.int64)
    train_idx, test_idx = train_test_split(indices, test_size=test_size, random_state=random_state, shuffle=True)
    print(f"Training samples: {len(train_idx)}")
    print(f"Test samples: {len(test_idx)}")
    return train_idx, test_idx


class IndexedFluxDataset(Dataset):
    def __init__(self, X, y, indices):
        self.X = X
        self.y = y
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        row = int(self.indices[idx])
        x = np.asarray(self.X[row], dtype=np.float32)
        y = np.asarray(self.y[row], dtype=np.float32)
        return torch.from_numpy(x).unsqueeze(-1), torch.from_numpy(y).unsqueeze(-1)


def create_dataloaders(X, y, train_indices, test_indices, batch_size, device, num_workers=0):
    train_dataset = IndexedFluxDataset(X, y, train_indices)
    test_dataset = IndexedFluxDataset(X, y, test_indices)

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, test_loader

def _empty_cache(device):
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()


def _torch_load_checkpoint(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def _remove_file_if_exists(path):
    if path and os.path.exists(path):
        os.remove(path)


def _format_elapsed(seconds):
    seconds = int(seconds)
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _save_best_checkpoint(path, model, optimizer, train_losses, test_losses, best_epoch, best_test_loss):
    checkpoint_dir = os.path.dirname(path)
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)
    torch.save(
        {
            "epoch": int(best_epoch),
            "model_epoch": int(best_epoch),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_losses": list(train_losses),
            "test_losses": list(test_losses),
            "best_test_loss": float(best_test_loss),
            "best_epoch": int(best_epoch),
        },
        path,
    )


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
    output_sample_ratio=1.0,
    checkpoint_path=None,
    best_checkpoint_path=None,
    patience=10
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

    start_epoch = 0
    best_test_loss = float("inf")
    best_epoch = -1
    early_stopped = False
    epochs_no_improve = 0
    train_losses, test_losses = [], []

    if patience is not None:
        patience = int(patience)
        if patience < 0:
            raise ValueError("patience must be nonnegative or None")

    if checkpoint_path and best_checkpoint_path is None:
        checkpoint_root, checkpoint_ext = os.path.splitext(checkpoint_path)
        best_checkpoint_path = f"{checkpoint_root}_best_tmp{checkpoint_ext}"

    if (
        checkpoint_path
        and best_checkpoint_path
        and os.path.abspath(best_checkpoint_path) == os.path.abspath(checkpoint_path)
    ):
        raise ValueError("best_checkpoint_path must be different from checkpoint_path")

    if best_checkpoint_path:
        _remove_file_if_exists(best_checkpoint_path)

    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Found checkpoint at {checkpoint_path}. Trying to resume training.")
        checkpoint = _torch_load_checkpoint(checkpoint_path, device)

        try:
            model.load_state_dict(checkpoint["model_state_dict"])
        except Exception as err:
            print(f"Could not load model weights from checkpoint. Starting fresh. Error: {err}")
        else:
            if "optimizer_state_dict" in checkpoint:
                try:
                    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                except Exception as err:
                    print(f"Could not load optimizer state. Continuing with fresh optimizer. Error: {err}")

            checkpoint_has_best_state = "model_epoch" in checkpoint or "best_epoch" in checkpoint
            start_epoch = int(checkpoint.get("model_epoch", checkpoint.get("epoch", 0)))
            train_losses = list(checkpoint.get("train_losses", []))
            test_losses = list(checkpoint.get("test_losses", []))
            train_losses = train_losses[:start_epoch]
            test_losses = test_losses[:start_epoch]

            if test_losses:
                history_best_epoch = int(np.argmin(test_losses) + 1)
                history_best_loss = float(min(test_losses))
                if checkpoint_has_best_state:
                    best_test_loss = float(checkpoint.get("best_test_loss", history_best_loss))
                    best_epoch = int(checkpoint.get("best_epoch", history_best_epoch))
                else:
                    best_test_loss = float(test_losses[-1])
                    best_epoch = int(start_epoch)
                    if history_best_epoch != start_epoch:
                        print(
                            "Existing checkpoint predates best-epoch metadata; "
                            f"using loaded epoch {start_epoch} as the resumable best state."
                        )

                if best_checkpoint_path and best_epoch == start_epoch:
                    _save_best_checkpoint(
                        best_checkpoint_path,
                        model,
                        optimizer,
                        train_losses,
                        test_losses,
                        best_epoch,
                        best_test_loss,
                    )

            print(f"Resumed from epoch {start_epoch}. Target epoch: {num_epochs}.")

    if start_epoch >= num_epochs:
        print(
            f"Checkpoint already trained to epoch {start_epoch}, "
            f"which is >= requested num_epochs={num_epochs}. Skipping training."
        )
        train_meta = {
            "best_test_loss": float(best_test_loss) if best_test_loss != float("inf") else None,
            "best_epoch": int(best_epoch),
            "elapsed_sec": 0.0,
            "num_epochs": int(num_epochs),
            "output_sample_ratio": float(output_sample_ratio),
            "start_epoch": int(start_epoch),
            "end_epoch": int(start_epoch),
            "model_epoch": int(best_epoch if best_epoch > 0 else start_epoch),
            "training_end_epoch": int(start_epoch),
            "resumed": bool(start_epoch > 0),
            "early_stopped": False,
            "patience": patience,
        }
        _remove_file_if_exists(best_checkpoint_path)
        return train_losses, test_losses, model, optimizer, train_meta

    total_outputs = len(out_indices)
    print(f"Early stopping patience: {patience}")

    for epoch in range(start_epoch, num_epochs):
        model.train()
        epoch_train_loss = 0.0

        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
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
                batch_X = batch_X.to(device, non_blocking=True)
                batch_y = batch_y.to(device, non_blocking=True)
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

        epoch_elapsed = _format_elapsed(time.time() - start_time)
        current_time = time.strftime("%H:%M")
        print(
            f"Epoch {epoch+1}/{num_epochs} | "
            f"Train Loss: {epoch_train_loss:.6f} | "
            f"Test Loss: {epoch_test_loss:.6f} | "
            f"Elapsed: {epoch_elapsed} | Time: {current_time}"
        )

        if epoch_test_loss < best_test_loss:
            best_test_loss = epoch_test_loss
            best_epoch = epoch + 1
            epochs_no_improve = 0
            if best_checkpoint_path:
                _save_best_checkpoint(
                    best_checkpoint_path,
                    model,
                    optimizer,
                    train_losses,
                    test_losses,
                    best_epoch,
                    best_test_loss,
                )
        else:
            epochs_no_improve += 1
            if (
                patience is not None
                and epochs_no_improve >= patience
            ):
                early_stopped = True
                print(
                    f"Early stopping at epoch {epoch+1}; "
                    f"best test loss {best_test_loss:.6f} was at epoch {best_epoch}."
                )
                break

        _empty_cache(device)
        gc.collect()

    training_end_epoch = len(test_losses)

    if best_checkpoint_path and os.path.exists(best_checkpoint_path):
        best_checkpoint = _torch_load_checkpoint(best_checkpoint_path, device)
        model.load_state_dict(best_checkpoint["model_state_dict"])
        if "optimizer_state_dict" in best_checkpoint:
            optimizer.load_state_dict(best_checkpoint["optimizer_state_dict"])
        train_losses = list(best_checkpoint.get("train_losses", train_losses))
        test_losses = list(best_checkpoint.get("test_losses", test_losses))
        _remove_file_if_exists(best_checkpoint_path)
        print(f"Restored best model state from epoch {best_epoch}.")

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
        "start_epoch": int(start_epoch),
        "end_epoch": int(training_end_epoch),
        "model_epoch": int(best_epoch),
        "training_end_epoch": int(training_end_epoch),
        "resumed": bool(start_epoch > 0),
        "early_stopped": bool(early_stopped),
        "patience": patience,
    }
    return train_losses, test_losses, model, optimizer, train_meta


def plot_loss_curves(train_losses, test_losses, d_model, n_heads, n_layers, d_ff, save_path=None, log_scale=True):
    epochs = np.arange(1, len(train_losses) + 1)
    plt.figure(figsize=(14, 10))
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.plot(epochs, train_losses, label="Training Loss")
    plt.plot(epochs, test_losses, label="Test Loss")
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
    num_epochs = 100
    patience = 10
    learning_rate = 1e-4
    dropout = 0.02
    output_sample_ratio = 1.0
    cache_dir = "./tmp_memmap"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    X = y = None
    train_loader = test_loader = None
    train_indices = test_indices = None
    try:
        X, y, inputs, outputs, input_token_indices, out_indices = load_data(DATA_PATH, cache_dir=cache_dir)
        train_indices, test_indices = prepare_split_indices(len(X), test_size=0.2, random_state=42)
        train_loader, test_loader = create_dataloaders(
            X,
            y,
            train_indices,
            test_indices,
            batch_size=batch_size,
            device=device,
            num_workers=0,
        )

        today = date.today().isoformat()
        model_name = f"{MODEL_NAME}_d{d_model}_h{n_heads}_l{n_layers}_ff{d_ff}"
        pic_dir = f"./pics/{today}/{model_name}"
        os.makedirs(pic_dir, exist_ok=True)

        model_save_dir = f"./models/{model_name}"
        checkpoint_path = f"{model_save_dir}/{model_name}_checkpoint.pth"

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
            output_sample_ratio=output_sample_ratio,
            checkpoint_path=checkpoint_path,
            patience=patience
        )

        model_save_path = f"{model_save_dir}/{model_name}.pth"
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        torch.save(model.state_dict(), model_save_path)
        model_epoch = int(train_meta.get("model_epoch", train_meta.get("best_epoch", num_epochs)))
        training_end_epoch = int(train_meta.get("training_end_epoch", train_meta.get("end_epoch", model_epoch)))
        print(f"\nBest-epoch model weights saved to {model_save_path} (epoch {model_epoch})")


        checkpoint = {
            "epoch": model_epoch,
            "model_epoch": model_epoch,
            "training_end_epoch": training_end_epoch,
            "best_epoch": int(train_meta.get("best_epoch", model_epoch)),
            "best_test_loss": train_meta.get("best_test_loss"),
            "early_stopped": bool(train_meta.get("early_stopped", False)),
            "patience": train_meta.get("patience", patience),
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
                "num_epochs": model_epoch,
                "requested_num_epochs": num_epochs,
                "training_end_epoch": training_end_epoch,
                "patience": patience,
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
                "n_train": int(len(train_indices)),
                "n_test": int(len(test_indices)),
                "input_token_indices": [int(i) for i in input_token_indices],
                "out_indices": [int(i) for i in out_indices],
            },
        }

        torch.save(checkpoint, checkpoint_path)
        print(f"Full checkpoint saved to {checkpoint_path}")

        plot_loss_curves(
            train_loss, test_loss, 
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            save_path=None
            #save_path=f"{pic_dir}/training_curve.png"
        )
    finally:
        train_loader = None
        test_loader = None
        X = None
        y = None
        gc.collect()
        removed_files = cleanup_memmap_files(DATA_PATH, cache_dir=cache_dir)
        if removed_files:
            print("Removed memmap cache files:")
            for path in removed_files:
                print(f"  - {path}")
