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
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import r2_score, mean_absolute_error

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
import seaborn as sns

from flux_transformer import FluxTransformer
from ecoli_iML1515_reactions import inputs, exchange_reactions

DATA_PATH = "./data/2025-09-10_iML1515_EX_training_data_49586_samples.csv"

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
    print(f"Number of output targets: {len(exchange_reactions)}")

    X = df[inputs].to_numpy(dtype=np.float32)
    y = df[[f"{col}_flux" for col in exchange_reactions]].to_numpy(dtype=np.float32)

    X_combined = np.hstack([X, np.zeros_like(y)])
    y_combined = np.hstack([np.zeros_like(X), y])

    return X_combined, y_combined, inputs, exchange_reactions

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

def train_model(d_model=128, n_heads=8, n_layers=3, d_ff=1024, num_epochs=100, learning_rate=0.001, dropout=0.02):
    start_time = time.time()

    model = FluxTransformer(
        vocab_size=len(inputs) + len(exchange_reactions),
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

def plot_prediction_sample(X_test, y_test, model, save_path=None):
    X_test_ = X_test.unsqueeze(-1)
    y_test_ = y_test.unsqueeze(-1)
    j = np.random.randint(0, X_test_.size(0), 1)[0]
    pred = model(X_test_[j].unsqueeze(0))
    true = y_test_[j].unsqueeze(0)
    plt.plot(pred[0,:,0].cpu().detach().numpy(), label='Predicted')
    plt.plot(true[0,:,0].cpu().detach().numpy(), label='True')
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

if __name__ == "__main__":
    set_seed()
    
    d_model = 64
    n_heads = 4
    n_layers = 2
    d_ff = 1024
    batch_size = 16
    num_epochs = 20
    learning_rate = 1e-4
    dropout = 0.02
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    X, y, input_cols, output_cols = load_data(DATA_PATH)
    X_train, X_test, y_train, y_test = prepare_tensors(X, y, device=device)
    train_loader, test_loader = create_dataloaders(X_train, y_train, X_test, y_test, batch_size)

    print_gpu_memory()

    train_loss, test_loss, model, optimizer = train_model(d_model, n_heads, n_layers, d_ff, num_epochs, learning_rate, dropout)

    today = date.today().isoformat()
    model_name = f"ecoli_iML1515_EX_d{d_model}_h{n_heads}_l{n_layers}_ff{d_ff}"
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
            'vocab_size': len(inputs) + len(exchange_reactions),
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