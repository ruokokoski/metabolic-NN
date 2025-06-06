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

datafile = "./data/simple_training_data_943_samples.csv"

def load_and_preprocess_data(filename):
    """Load and preprocess the training data"""
    df = pd.read_csv(filename)
    print(f"\nLoaded data with {len(df)} samples from {filename}")
    print(df[['glucose_uptake', 'oxygen_uptake']].describe())
    
    # Inputs: glucose and oxygen uptake rates
    X = df[['glucose_uptake', 'oxygen_uptake']].values
    # Output: biomass flux (can be extended to multiple outputs)
    y = df['Biomass_Ecoli_core_flux'].values.reshape(-1, 1)
    
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
    def __init__(self, input_size=2, hidden_size=64, output_size=1):
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
    
def plot_actual_vs_predicted(y_true, y_pred, title, filename):
    """Scatter plot of actual vs predicted values"""
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.2, s=10)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'k--', lw=2)
    plt.xlabel('Actual Biomass Flux')
    plt.ylabel('Predicted Biomass Flux')
    plt.title(title)
    plt.grid(True)
    plt.savefig(filename)
    plt.close()

def plot_residuals(y_true, y_pred, title, filename):
    """Plot residuals (errors) vs actual values"""
    residuals = y_true - y_pred
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='-')
    plt.xlabel('Actual Biomass Flux')
    plt.ylabel('Residuals (Actual - Predicted)')
    plt.title(title)
    plt.grid(True)
    plt.savefig(filename)
    plt.close()

def plot_error_distribution(y_true, y_pred, title, filename):
    """Histogram of prediction errors"""
    errors = y_true - y_pred
    plt.figure(figsize=(8, 6))
    sns.histplot(errors, kde=True, legend=False)
    plt.xlabel('Prediction Error')
    plt.ylabel('Frequency')
    plt.title(title)
    plt.grid(True)
    plt.savefig(filename)
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

epochs = 100
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

    if (epoch+1) % 10 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {loss.item():.4f}, Test Loss: {test_loss:.4f}")

with torch.no_grad():
    test_preds_scaled = model(X_test_tensor).numpy()
    test_preds = y_scaler.inverse_transform(test_preds_scaled)
    test_true = y_scaler.inverse_transform(y_test_tensor.numpy())

r2 = r2_score(test_true, test_preds)
print(f"R² Score (Test): {r2:.4f}")

# 1. Plot training curves
plot_loss_curves(train_losses, test_losses, './pics/training_curve.png')

# 2. Plot Actual vs Predicted (Test set)
plot_actual_vs_predicted(test_true, test_preds, 'Actual vs Predicted Biomass Flux on Test Set', './pics/true_vs_predicted_test.png')

# 3. Residual plot (Test set)
plot_residuals(test_true, test_preds, 
               'Residuals vs True Values (Test Set)',
               './pics/residuals_test.png')

# 4. Error distribution (Test set)
plot_error_distribution(test_true, test_preds, 
                        'Prediction Error Distribution (Test Set)',
                        './pics/error_distribution_test.png')

# 5. Plot feature importance
plot_feature_importance(model, ['Glucose Uptake', 'Oxygen Uptake'])

torch.save(model.state_dict(), "./models/metabolic_nn.pth")
import joblib
joblib.dump(x_scaler, "./models/input_scaler.pkl")
joblib.dump(y_scaler, "./models/output_scaler.pkl")

print("Model and scalers saved.")


