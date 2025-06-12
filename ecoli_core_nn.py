import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns

datafile = "./data/training_data_18264_samples.csv"

def load_and_preprocess_data(filename):
    """Load and preprocess the training data"""
    df = pd.read_csv(filename)
    print(f"\nLoaded data with {len(df)} samples from {filename}")
    print(df[['glucose_uptake', 'oxygen_uptake', 'ammonia_uptake', 'phosphate_uptake']].describe())
    print(df[['EX_co2_e_flux', 'EX_h2o_e_flux', 'EX_h_e_flux', 'Biomass_Ecoli_core_flux']].describe())
    print()
    
    input_cols = [
        'glucose_uptake',
        'oxygen_uptake',
        'ammonia_uptake',
        'phosphate_uptake',

    ]
    output_cols = [
        'EX_co2_e_flux',
        'EX_h2o_e_flux',
        'EX_h_e_flux',
        'Biomass_Ecoli_core_flux'
    ]

    X = df[input_cols].values.astype(np.float32)
    y = df[output_cols].values.astype(np.float32)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Normalize inputs and outputs
    x_scaler = StandardScaler().fit(X_train)
    y_scaler = StandardScaler().fit(y_train)

    X_train = x_scaler.transform(X_train)
    X_test = x_scaler.transform(X_test)
    y_train = y_scaler.transform(y_train)
    y_test = y_scaler.transform(y_test)

    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

    return X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, x_scaler, y_scaler

class MetabolicNN(nn.Module):
    """Neural network to predict metabolic fluxes"""
    def __init__(self, input_size=4, hidden_size=256, output_size=4):
        super(MetabolicNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
    
    def forward(self, x):
        return self.model(x)
    
def plot_loss_curves(train_losses, test_losses, save_path="./pics/training_curve.png"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Training Loss")
    plt.plot(test_losses, label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Training and Test Loss")
    plt.legend()
    plt.savefig(save_path)
    plt.close()
    print(f"\nTraining curve saved to {save_path}")
    
def plot_diagnostics_2x2(y_true, y_pred, label, save_path):
    """Creates a 2x2 matrix of plots: actual vs predicted, residuals, error distribution, and histogram of actuals"""
    residuals = y_true - y_pred
    errors = residuals

    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    
    # Actual vs Predicted
    axs[0, 0].scatter(y_true, y_pred, alpha=0.2, s=10)
    axs[0, 0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'k--', lw=2)
    axs[0, 0].set_title(f'Actual vs Predicted: {label}')
    axs[0, 0].set_xlabel('Actual')
    axs[0, 0].set_ylabel('Predicted')
    axs[0, 0].grid(True)

    # Residuals plot
    axs[0, 1].scatter(y_true, residuals, alpha=0.5)
    axs[0, 1].axhline(y=0, color='r', linestyle='-')
    axs[0, 1].set_title(f'Residuals: {label}')
    axs[0, 1].set_xlabel('Actual')
    axs[0, 1].set_ylabel('Residuals')
    axs[0, 1].grid(True)

    # Error distribution
    sns.histplot(errors, kde=True, ax=axs[1, 0], legend=False)
    axs[1, 0].set_title(f'Prediction Error Distribution: {label}')
    axs[1, 0].set_xlabel('Prediction Error')
    axs[1, 0].set_ylabel('Frequency')
    axs[1, 0].grid(True)

    # Histogram of actual values
    sns.histplot(y_true, kde=True, ax=axs[1, 1], color='g', legend=False)
    axs[1, 1].set_title(f'Actual Value Distribution: {label}')
    axs[1, 1].set_xlabel('Actual Value')
    axs[1, 1].set_ylabel('Frequency')
    axs[1, 1].grid(True)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()

def plot_feature_importance(model, feature_names):
    """Visualize feature importance using first-layer weights"""
    weights = model.model[0].weight.data.numpy()
    importance = np.mean(np.abs(weights), axis=0)
    
    plt.figure(figsize=(8, 5))
    plt.bar(feature_names, importance)
    plt.xlabel('Features')
    plt.ylabel('Average Absolute Weight')
    plt.title('Feature Importance from First Layer Weights')
    plt.savefig('./pics/feature_importance.png')
    plt.close()

X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, x_scaler, y_scaler = load_and_preprocess_data(datafile)

model = MetabolicNN()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Create models directory if it doesn't exist
os.makedirs("./models", exist_ok=True)

# Train the model
train_losses = []
test_losses = []

epochs = 500
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()

    train_losses.append(loss.item())

    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test_tensor)
        test_loss = criterion(test_outputs, y_test_tensor).item()
        test_losses.append(test_loss)

    if (epoch+1) % 100 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {loss.item():.4f}, Test Loss: {test_loss:.4f}")

with torch.no_grad():
    test_preds_scaled = model(X_test_tensor).numpy()
    test_preds = y_scaler.inverse_transform(test_preds_scaled)
    test_true = y_scaler.inverse_transform(y_test_tensor.numpy())

output_labels = ['EX_co2_e_flux', 'EX_h2o_e_flux', 'EX_h_e_flux', 'Biomass_Ecoli_core_flux']

# Plot training curves
plot_loss_curves(train_losses, test_losses, './pics/training_curve.png')

for i, label in enumerate(output_labels):
    actual = test_true[:, i]
    predicted = test_preds[:, i]

    plot_diagnostics_2x2(actual, predicted,
                         label,
                         f'./pics/diagnostics_{label}.png')

# 5. Plot feature importance
plot_feature_importance(model, ['Glucose Uptake', 'Oxygen Uptake', 'Ammonia Uptake', 'Phosphate Uptake'])

torch.save(model.state_dict(), "./models/metabolic_nn.pth")
import joblib
joblib.dump(x_scaler, "./models/input_scaler.pkl")
joblib.dump(y_scaler, "./models/output_scaler.pkl")

print("Model and scalers saved.")

for i, label in enumerate(output_labels):
    r2 = r2_score(test_true[:, i], test_preds[:, i])
    print(f"{label}: R² = {r2:.4f}")

