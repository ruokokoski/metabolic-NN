import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt

def load_and_preprocess_data(filename):
    """Load and preprocess the training data"""
    df = pd.read_csv(filename)
    print(f"\nLoaded data with {len(df)} samples from {filename}")
    
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
    X_train_tensor = torch.tensor(X_train)
    y_train_tensor = torch.tensor(y_train)
    X_test_tensor = torch.tensor(X_test)
    y_test_tensor = torch.tensor(y_test)

    return X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, x_scaler, y_scaler

class MetabolicNN(nn.Module):
    """Neural network to predict metabolic fluxes"""
    def __init__(self, input_size=2, hidden_size=64, output_size=3):
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
    

filename = "./data/simple_training_data_9514_samples.csv"
X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, x_scaler, y_scaler = load_and_preprocess_data(filename)

model = MetabolicNN()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
epochs = 100
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()

    if (epoch+1) % 10 == 0:
        model.eval()
        test_preds = model(X_test_tensor)
        test_loss = criterion(test_preds, y_test_tensor).item()
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {loss.item():.4f}, Test Loss: {test_loss:.4f}")

torch.save(model.state_dict(), "./models/metabolic_nn.pth")
import joblib
joblib.dump(x_scaler, "./models/input_scaler.pkl")
joblib.dump(y_scaler, "./models/output_scaler.pkl")

print("Model and scalers saved.")
