import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import flwr as fl
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score
from model import get_model

def load_data(csv_path):
    df = pd.read_csv(csv_path)
    X = df.iloc[:, :-1].values.astype(np.float32)
    y = df.iloc[:, -1].values.astype(np.int64)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    split_idx = int(len(X) * 0.8)
    train_dataset = TensorDataset(torch.tensor(X[:split_idx]), torch.tensor(y[:split_idx]))
    val_dataset = TensorDataset(torch.tensor(X[split_idx:]), torch.tensor(y[split_idx:]))
    return DataLoader(train_dataset, batch_size=256, shuffle=True), DataLoader(val_dataset, batch_size=256)

class SDNClient(fl.client.NumPyClient):
    def __init__(self, client_id):
        self.client_id = client_id
        self.model = get_model()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        csv_path = f"datasets/sw{client_id}.csv"
        self.trainloader, self.valloader = load_data(csv_path)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0005)
        self.current_round = 0
    
    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]
    
    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = dict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)
    
    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.current_round += 1
        for epoch in range(5):
            for batch in self.trainloader:
                inputs, labels = batch[0].to(self.device), batch[1].to(self.device)
                self.optimizer.zero_grad()
                loss = self.criterion(self.model(inputs), labels)
                loss.backward()
                self.optimizer.step()
                
        if self.current_round == 20:
            torch.save(self.model.state_dict(), 'global_model.pth')
            print(f"\n✅ Round {self.current_round}: Global model saved!")
            
        return self.get_parameters({}), len(self.trainloader.dataset), {}
    
    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        correct, total, loss = 0, 0, 0.0
        all_preds, all_labels = [], []
        self.model.eval()
        
        with torch.no_grad():
            for batch in self.valloader:
                inputs, labels = batch[0].to(self.device), batch[1].to(self.device)
                outputs = self.model(inputs)
                loss += self.criterion(outputs, labels).item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Loss Fix: ব্যাচ সংখ্যা দিয়ে ভাগ করা হয়েছে
        avg_loss = loss / len(self.valloader)
        accuracy = correct / total
        
        # নতুন মেট্রিক্স: Precision, Recall, F1-Score
        prec = precision_score(all_labels, all_preds, average='macro', zero_division=0)
        rec = recall_score(all_labels, all_preds, average='macro', zero_division=0)
        f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        
        return float(avg_loss), total, {"accuracy": accuracy, "precision": prec, "recall": rec, "f1": f1}

if __name__ == "__main__":
    import sys
    client_id = sys.argv[1] if len(sys.argv) > 1 else "1"
    fl.client.start_client(server_address="127.0.0.1:8080", client=SDNClient(client_id).to_client())
