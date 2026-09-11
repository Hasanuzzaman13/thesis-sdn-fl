import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import flwr as fl
from sklearn.metrics import accuracy_score

class IntrusionDetectionNet(nn.Module):
    def __init__(self):
        super(IntrusionDetectionNet, self).__init__()
        self.fc1 = nn.Linear(4, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

def load_data(client_id):
    file_path = f"datasets/sw{client_id}.csv"
    df = pd.read_csv(file_path)
    y = df['Label'].values
    X = df.drop('Label', axis=1).values
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    dataset = TensorDataset(X_tensor, y_tensor)
    return DataLoader(dataset, batch_size=32, shuffle=True)

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, client_id):
        self.net = IntrusionDetectionNet()
        self.client_id = client_id
        self.trainloader = load_data(client_id)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.net.parameters(), lr=0.01)

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.net.train()
        for epoch in range(2): 
            for batch_x, batch_y in self.trainloader:
                self.optimizer.zero_grad()
                outputs = self.net(batch_x)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
        return self.get_parameters(config={}), len(self.trainloader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.net.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for batch_x, batch_y in self.trainloader:
                outputs = self.net(batch_x)
                loss = self.criterion(outputs, batch_y)
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(batch_y.cpu().numpy())
        
        avg_loss = total_loss / len(self.trainloader)
        accuracy = accuracy_score(all_labels, all_preds)
        return avg_loss, len(self.trainloader.dataset), {"accuracy": float(accuracy)}

def client_fn(cid: str) -> fl.client.Client:
    client_id = int(cid) + 1 
    return FlowerClient(client_id=client_id).to_client()

print("🚀 Starting Federated Learning Simulation...")
print("4 clients (sw1, sw2, sw3, sw4) are connecting to the server...\n")

fl.simulation.start_simulation(
    client_fn=client_fn,
    num_clients=4,
    config=fl.server.ServerConfig(num_rounds=3),
)

print("\n🎉 Federated Learning Training completed successfully!")
