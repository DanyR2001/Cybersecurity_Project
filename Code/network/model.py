import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler

class EmberMalwareNet(nn.Module):
    """
    Rete neurale per malware detection su dataset EMBER.
    EMBER ha 2381 features estratte da file PE Windows.
    Architettura ottimizzata basata su embernn.py (F1=0.9)
    """
    def __init__(self, input_dim=2381, dropout_rate=0.5):  # Dropout aumentato a 0.5
        super(EmberMalwareNet, self).__init__()
        
        # StandardScaler per normalizzazione (come embernn.py)
        self.scaler = StandardScaler()
        self.is_fitted = False
        
        # Architettura wide come embernn.py: 4000 -> 2000 -> 100
        self.fc1 = nn.Linear(input_dim, 4000)
        self.bn1 = nn.BatchNorm1d(4000)
        self.dropout1 = nn.Dropout(dropout_rate)
        
        self.fc2 = nn.Linear(4000, 2000)
        self.bn2 = nn.BatchNorm1d(2000)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        self.fc3 = nn.Linear(2000, 100)
        self.bn3 = nn.BatchNorm1d(100)
        self.dropout3 = nn.Dropout(dropout_rate)
        
        # Output layer per classificazione binaria
        self.output = nn.Linear(100, 1)
        
        self._init_weights()

    def _init_weights(self):
        # inizializzazione consigliata per ReLU
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        # x assumed float tensor already normalized and on device
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)

        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)

        x = F.relu(self.bn3(self.fc3(x)))
        x = self.dropout3(x)

        logits = self.output(x)  # shape [batch, 1]
        return torch.sigmoid(logits).squeeze(1)

# Funzione per training
def train_model(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for data, target in train_loader:
        data = data.to(device).float()
        target = target.to(device).float()

        optimizer.zero_grad()
        logits = model(data)            # logits shape [batch]
        loss = criterion(logits, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * data.size(0)

        preds = (torch.sigmoid(logits) > 0.5).long()
        correct += (preds == target.long()).sum().item()
        total += data.size(0)

    return total_loss / total, correct / total


def validate_model(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in val_loader:
            data = data.to(device).float()
            target = target.to(device).float()

            logits = model(data)
            loss = criterion(logits, target)
            total_loss += loss.item() * data.size(0)

            preds = (torch.sigmoid(logits) > 0.5).long()
            correct += (preds == target.long()).sum().item()
            total += data.size(0)

    return total_loss / total, correct / total

# Esempio di utilizzo
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model = EmberMalwareNet(input_dim=2381).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    # Nota: normalizza X_train esternamente con sklearn.StandardScaler BEFORE creating tensors/dataloader