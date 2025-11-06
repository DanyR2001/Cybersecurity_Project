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
        
    def fit_scaler(self, X):
        """Fit dello scaler sui dati di training"""
        if not self.is_fitted:
            self.scaler.fit(X)
            self.is_fitted = True
    
    def transform(self, X):
        """Applica la normalizzazione"""
        if self.is_fitted:
            return self.scaler.transform(X)
        return X
        
    def forward(self, x):
        # Layer 1
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        
        # Layer 2
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        
        # Layer 3
        x = self.fc3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.dropout3(x)
        
        # Output con sigmoid (CRITICO per F1 score)
        x = self.output(x)
        return torch.sigmoid(x)  # Sigmoid aggiunto nel forward


# Funzione per training
def train_model(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)  # Già con sigmoid nel forward
        loss = criterion(output.squeeze(), target.float())
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pred = (output > 0.5).float()  # No sigmoid qui, già applicato
        correct += (pred.squeeze() == target).sum().item()
        total += target.size(0)
    
    return total_loss / len(train_loader), correct / total


# Funzione per validazione
def validate_model(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)  # Già con sigmoid nel forward
            loss = criterion(output.squeeze(), target.float())
            
            total_loss += loss.item()
            pred = (output > 0.5).float()  # No sigmoid qui, già applicato
            correct += (pred.squeeze() == target).sum().item()
            total += target.size(0)
    
    return total_loss / len(val_loader), correct / total


# Esempio di utilizzo
if __name__ == "__main__":
    # Setup - supporto per M1/M2/M3 Mac (MPS), CUDA e CPU
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Silicon GPU (MPS)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA GPU")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    
    model = EmberMalwareNet(input_dim=2381, dropout_rate=0.5).to(device)
    
    # IMPORTANTE: Ora usa BCELoss invece di BCEWithLogitsLoss
    # perché sigmoid è già nel forward
    criterion = nn.BCELoss()
    
    # Optimizer SGD come embernn.py (migliore di Adam per EMBER)
    optimizer = torch.optim.SGD(
        model.parameters(), 
        lr=0.1,           # LR alto come embernn.py
        momentum=0.9,     # Momentum alto
        weight_decay=1e-6 # Weight decay come embernn.py
    )
    
    # Scheduler opzionale (embernn.py non lo usa ma può aiutare)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # Training loop (esempio con dati fittizi)
    print(f"Modello creato con {sum(p.numel() for p in model.parameters())} parametri")
    print(f"Device: {device}")
    print("\nNOTE IMPORTANTI:")
    print("1. Usa 10 epochs invece di 30 per evitare overfitting")
    print("2. Usa batch_size=512 invece di 256")
    print("3. Normalizza i dati con model.fit_scaler(X_train) prima del training")
    print("4. Usa BCELoss invece di BCEWithLogitsLoss")
    
    # Esempio di come usare il modello:
    # 
    # # 1. Carica dati
    # X_train, y_train = ember.read_vectorized_features(...)
    # 
    # # 2. FIT dello scaler
    # model.fit_scaler(X_train)
    # X_train_scaled = model.transform(X_train)
    # 
    # # 3. Crea DataLoader con dati normalizzati
    # train_dataset = TensorDataset(
    #     torch.FloatTensor(X_train_scaled),
    #     torch.FloatTensor(y_train)
    # )
    # train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
    # 
    # # 4. Training loop (10 epochs)
    # for epoch in range(10):
    #     train_loss, train_acc = train_model(model, train_loader, criterion, optimizer, device)
    #     val_loss, val_acc = validate_model(model, val_loader, criterion, device)
    #     scheduler.step(val_loss)
    #     print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
    #           f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")