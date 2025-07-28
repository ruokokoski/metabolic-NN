# Difference to v04: identify constant and zero inflated fluxes

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
from datetime import date

datafile = "./data/2025-07-15_full_training_data_98066_samples.csv" # carbons log-uniform, others uniform
#datafile = "./data/2025-07-14_full_training_data_99973_easy_samples.csv"

class MetabolicNN(nn.Module):
    """Neural network to predict metabolic fluxes"""
    def __init__(self, input_size=20, hidden_size=512, output_size=95):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.LeakyReLU(0.01),

            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.LeakyReLU(0.01),

            nn.Linear(hidden_size, output_size)
        )
    
    def forward(self, x):
        return self.model(x)

def load_data(filename):
    """Load and preprocess the training data"""
    input_cols = [
        'EX_glc__D_e', 'EX_fru_e', 'EX_lac__D_e', 'EX_pyr_e', 'EX_ac_e', 'EX_akg_e', 
        'EX_succ_e', 'EX_fum_e', 'EX_mal__L_e', 'EX_etoh_e', 'EX_acald_e', 'EX_for_e',
        'EX_gln__L_e', 'EX_glu__L_e',
        'EX_co2_e', 'EX_h_e', 'EX_h2o_e', 'EX_nh4_e', 'EX_o2_e', 'EX_pi_e',
    ]

    output_cols = [
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
    
    df = pd.read_csv(filename)

    # Fill missing inputs with 0 (i.e., not uptaken)
    df[input_cols] = df[input_cols].fillna(0)

    print(f"\nLoaded data with {len(df)} samples from {filename}")
    print(f"Total outputs: {len(output_cols)}")

    X = df[input_cols].values.astype(np.float32)
    y = df[output_cols].values.astype(np.float32)

    return X, y, input_cols, output_cols

def identify_zero_inflated_fluxes(y_train, output_cols, threshold=0.01):
    problematic_indices = []
    for i, col in enumerate(output_cols):
        if col.startswith('EX_'):
            continue
        zero_ratio = np.mean(np.abs(y_train[:, i]) < threshold)
        if zero_ratio > 0.5:
            problematic_indices.append(i)
    print(f"Fluxes with >50% near-zero values\n{[output_cols[i] for i in problematic_indices]}")
    print(f"{len(problematic_indices)} in total")
    return problematic_indices

def track_gradient_norms(model):
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5
    
def plot_loss_curves(train_losses, test_losses, save_path, log_scale=True):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(14, 10))
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.plot(train_losses, label="Training Loss")
    plt.plot(test_losses, label="Test Loss")
    if log_scale:
        plt.yscale('log')
    plt.xlabel("Epoch", fontsize=18)
    plt.ylabel("Loss", fontsize=18)
    plt.title("Training and Test Loss", fontsize=20)
    plt.grid(True)
    plt.legend(fontsize=16)
    plt.savefig(save_path)
    plt.close()
    #print(f"\nTraining curve saved to {save_path}")
    
def plot_diagnostics_2x2(y_true, y_pred, label, save_path):
    """Creates a 2x2 matrix of plots: true vs predicted, residuals, error distribution, and histogram of actuals"""
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
    sns.histplot(residuals, kde=True, ax=axs[1, 0], legend=False, color='indianred')
    axs[1, 0].set_title(f'Prediction Error Distribution: {label}')
    axs[1, 0].set_xlabel('Prediction Error')
    axs[1, 0].set_ylabel('Frequency')
    axs[1, 0].grid(True)

    # Histogram of actual values
    sns.histplot(y_true, kde=True, bins=100, ax=axs[1, 1], legend=False, color='mediumseagreen')
    axs[1, 1].set_title(f'True Value Distribution: {label}')
    axs[1, 1].set_xlabel('True Value')
    axs[1, 1].set_ylabel('Frequency')
    axs[1, 1].grid(True)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()

def save_individual_diagnostic_plots(y_true, y_pred, label, save_path):
    """Save individual diagnostic plots separately"""
    residuals = y_true - y_pred

    # True vs Predicted
    plt.figure(figsize=(14, 10))
    plt.scatter(y_true, y_pred, alpha=0.2, s=10, color='royalblue')
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'k--', lw=2)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlabel('True value', fontsize=18)
    plt.ylabel('Predicted', fontsize=18)
    plt.title(f'True vs Predicted: {label}', fontsize=20)
    plt.grid(True)
    plt.savefig(f"{save_path}_{label}_true_vs_pred.png")
    plt.close()

    # Residuals plot
    plt.figure(figsize=(14, 10))
    plt.scatter(y_true, residuals, alpha=0.2, color='darkorange')
    plt.axhline(y=0, color='r', linestyle='-')
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlabel('True value', fontsize=18)
    plt.ylabel('Residuals', fontsize=18)
    plt.title(f'Residuals: {label}', fontsize=20)
    plt.grid(True)
    plt.savefig(f"{save_path}_{label}_residuals.png")
    plt.close()

    # Error distribution
    plt.figure(figsize=(14, 10))
    sns.histplot(residuals, kde=True, color='indianred')
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlabel('Prediction Error', fontsize=18)
    plt.ylabel('Frequency', fontsize=18)
    plt.title(f'Error Distribution: {label}', fontsize=20)
    plt.grid(True)
    plt.savefig(f"{save_path}_{label}_error_dist.png")
    plt.close()

    # Histogram of actual values
    plt.figure(figsize=(14, 10))
    sns.histplot(y_true, kde=True, color='mediumseagreen')
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlabel('True value', fontsize=18)
    plt.ylabel('Frequency', fontsize=18)
    plt.title(f'True Value Distribution: {label}', fontsize=20)
    plt.grid(True)
    plt.savefig(f"{save_path}_{label}_true_dist.png")
    plt.close()

def plot_feature_importance(model, feature_names, save_path):
    """Visualize feature importance using first-layer weights"""
    weights = model.model[0].weight.data.numpy()
    importance = np.mean(np.abs(weights), axis=0)
    
    plt.figure(figsize=(14, 13))
    plt.xticks(rotation=45, ha='right', fontsize=14)
    plt.yticks(fontsize=16)
    plt.bar(feature_names, importance, color='seagreen')
    plt.xlabel('Features', fontsize=18)
    plt.ylabel('Average Absolute Weight', fontsize=18)
    plt.title('Feature Importance from First Layer Weights', fontsize=20)
    plt.savefig(save_path)
    plt.close()

def plot_gradient_norms(gradient_norms, save_path=None, log_scale=True):
    plt.figure(figsize=(14, 10))
    plt.plot(gradient_norms)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlabel("Epoch", fontsize=18)
    plt.ylabel("Gradient Norm (L2)", fontsize=18)
    plt.title("Gradient Norms Over Epochs", fontsize=20)
    if log_scale:
        plt.yscale('log')
    plt.grid(True)
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()
        #print(f"Gradient norm plot saved to {save_path}")
    else:
        plt.show()

def plot_standardized_errors(y_true, y_pred, output_labels, y_scaler, save_path):
    """Plot standardized errors across all outputs"""
    plt.figure(figsize=(14, 10))
    
    for i, label in enumerate(output_labels):
        # Calculate standardized residuals (original units / std of training data)
        std = y_scaler.scale_[i]
        residuals = (y_true[:, i] - y_pred[:, i]) / std
        
        sns.kdeplot(residuals, label=f"{label} (σ={std:.2f})", fill=True, alpha=0.15)
    
    plt.axvline(0, color='r', linestyle='--')
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlabel('Standardized Residuals (Original Units / σ)', fontsize=18)
    plt.ylabel('Density', fontsize=18)
    plt.title('Standardized Error Distributions Across Outputs', fontsize=20)
    plt.legend(fontsize=16)
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def preprocess_data(X, y, output_cols):
    """Handle constant outputs, scaling, and tensor conversion"""
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Identify constant outputs
    output_stats = pd.DataFrame(y_train, columns=output_cols).agg(['mean', 'std'])
    low_std_outputs = output_stats.columns[output_stats.loc['std'] <= 0.1]
    non_low_std_outputs = [col for col in output_cols if col not in low_std_outputs]
    
    # Get indices and values
    constant_indices = [output_cols.index(col) for col in low_std_outputs]
    non_constant_indices = [output_cols.index(col) for col in non_low_std_outputs]
    constant_values = output_stats.loc['mean', low_std_outputs].values.astype(np.float32)
    
    print(f"Non-constant outputs: {len(non_low_std_outputs)}")
    print(f"\nConstant outputs ({len(low_std_outputs)}):")
    print(low_std_outputs)
    print("Means of outputs with std ≤ 0.1:")
    print(constant_values)
    print()

    # Extract non-constant outputs
    y_train_non_constant = y_train[:, non_constant_indices]
    y_test_non_constant = y_test[:, non_constant_indices]
    
    # Identify zero-inflated fluxes
    filtered_output_cols = [output_cols[i] for i in non_constant_indices]
    zero_inflated_indices = identify_zero_inflated_fluxes(y_train_non_constant, filtered_output_cols)
    print(f'Zero-inflated indices: {zero_inflated_indices}')

    # Scale features and targets
    x_scaler = StandardScaler().fit(X_train)
    y_scaler = StandardScaler().fit(y_train_non_constant)
    X_train_scaled = x_scaler.transform(X_train)
    X_test_scaled = x_scaler.transform(X_test)
    y_train_scaled = y_scaler.transform(y_train_non_constant)
    y_test_scaled = y_scaler.transform(y_test_non_constant)

    # Convert to tensors
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test_scaled, dtype=torch.float32)
    
    return (
        X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor,
        constant_indices, non_constant_indices, low_std_outputs, constant_values,
        x_scaler, y_scaler, y_test
    )

def train_model(model, X_train, y_train, X_test, y_test, epochs=1000):
    """Train the model with GPU support if available"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = model.to(device)
    X_train = X_train.to(device)
    y_train = y_train.to(device)
    X_test = X_test.to(device)
    y_test = y_test.to(device)
    
    train_losses = []
    test_losses = []
    gradient_norms = []
    best_test_loss = float('inf')
    best_epoch = -1

    criterion = nn.HuberLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.0003, weight_decay=1e-5)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        grad_norm = track_gradient_norms(model)
        gradient_norms.append(grad_norm)

        optimizer.step()

        # Validation on test set
        model.eval()
        with torch.no_grad():
            train_loss = loss.item()
            test_outputs = model(X_test)
            test_loss = criterion(test_outputs, y_test).item()

        train_losses.append(train_loss)
        test_losses.append(test_loss)
        
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_epoch = epoch

        if (epoch+1) % 50 == 0:
            print(f"Epoch {epoch+1}/{epochs} \t | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f} | Grad Norm: {grad_norm:.4e}")

    print(f"Best test loss at epoch {best_epoch+1}: {best_test_loss:.4f}\n")
    
    # Move model and data back to CPU
    model = model.to('cpu')
    return model, train_losses, test_losses, gradient_norms


if __name__ == "__main__":
    X, y, input_cols, output_cols = load_data(datafile)

    # Preprocess data and prepare tensors
    (
        X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor,
        constant_indices, non_constant_indices, low_std_outputs, constant_values,
        x_scaler, y_scaler, y_test_raw
    ) = preprocess_data(X, y, output_cols)

    # Initialize model
    model = MetabolicNN(
        input_size=X_train_tensor.shape[1],
        output_size=len(non_constant_indices)
    )
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params:,}")

    print("\nTraining model:")
    model, train_losses, test_losses, gradient_norms = train_model(
        model=model,
        X_train=X_train_tensor,
        y_train=y_train_tensor,
        X_test=X_test_tensor,
        y_test=y_test_tensor,
        epochs=2200
    )

    # Evaluate model
    model.eval()
    with torch.no_grad():
        y_pred_scaled = model(X_test_tensor).numpy()
        y_pred = y_scaler.inverse_transform(y_pred_scaled)

        y_pred_full = np.zeros((len(X_test_tensor), len(output_cols)))
        y_pred_full[:, non_constant_indices] = y_pred
        y_pred_full[:, constant_indices] = constant_values
        y_true_full = y_test_raw

    today = date.today().isoformat()
    pic_dir = f"./pics/{today}"
    os.makedirs(pic_dir, exist_ok=True)
    model_name = "ecoli_core_v5"

    # Plot training curves
    plot_loss_curves(train_losses, test_losses, f'{pic_dir}/{model_name}_training_curve.png')

    for i, label in enumerate(output_cols):
        actual = y_true_full[:, i]
        predicted = y_pred_full[:, i]

        plot_diagnostics_2x2(actual, predicted,
                            label,
                            f'{pic_dir}/{model_name}_diagnostics_{label}.png')
        #save_individual_diagnostic_plots(actual, predicted, label, f'{pic_dir}/{model_name}')

    plot_feature_importance(model, input_cols, f'{pic_dir}/{model_name}_feature_importance.png')

    plot_gradient_norms(gradient_norms, f"{pic_dir}/{model_name}_gradient_norms.png")

    for i, label in enumerate(output_cols):
        actual = y_true_full[:, i]
        predicted = y_pred_full[:, i]
        if label in low_std_outputs:
            # Use Mean Absolute Error (MAE) for near-constant outputs (R2 numerically unstable)
            mae = np.mean(np.abs(actual - predicted))
            print(f"{label}: MAE = {mae:.4e} (near-constant output)")
        else:
            r2 = r2_score(actual, predicted)
            print(f"{label}: R² = {r2:.4f}")

    torch.save(model.state_dict(), f"./models/{model_name}_metabolic_nn.pth")
    import joblib
    joblib.dump(x_scaler, f"./models/{model_name}_input_scaler.pkl")
    joblib.dump(y_scaler, f"./models/{model_name}_output_scaler.pkl")

    print("\nModel and scalers saved.")
