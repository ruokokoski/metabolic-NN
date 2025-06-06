import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import os

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
    

filename = "./data/simple_training_data_9514_samples.csv"
X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, x_scaler, y_scaler = load_and_preprocess_data(filename)

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

# Plot training curves
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training Loss')
plt.plot(test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Training and Test Loss')
plt.legend()
plt.show()
plt.savefig('./models/training_curve.png')
print("\nTraining curve saved to ./models/training_curve.png")

torch.save(model.state_dict(), "./models/metabolic_nn.pth")
import joblib
joblib.dump(x_scaler, "./models/input_scaler.pkl")
joblib.dump(y_scaler, "./models/output_scaler.pkl")

print("Model and scalers saved.")


