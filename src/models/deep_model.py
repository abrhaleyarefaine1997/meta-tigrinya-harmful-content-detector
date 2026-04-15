import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class MLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.net(x)


class DeepModel:
    def __init__(self, input_dim, lr=1e-3, batch_size=32, epochs=10):
        self.input_dim = input_dim
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = MLP(input_dim).to(self.device)
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def _to_tensor(self, X, y=None):
        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        if y is not None:
            y = torch.tensor(y, dtype=torch.float32).view(-1, 1).to(self.device)
            return X, y
        return X

    def train(self, X, y):
        X, y = self._to_tensor(X, y)

        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model.train()

        for epoch in range(self.epochs):
            total_loss = 0

            for batch_X, batch_y in loader:
                self.optimizer.zero_grad()

                logits = self.model(batch_X)
                loss = self.criterion(logits, batch_y)

                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            print(f"Epoch {epoch+1}/{self.epochs} - Loss: {total_loss:.4f}")

    def predict_proba(self, X):
        self.model.eval()
        X = self._to_tensor(X)

        with torch.no_grad():
            logits = self.model(X)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()

        return probs

    def predict(self, X, threshold=0.5):
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(int)

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.to(self.device)
        return self