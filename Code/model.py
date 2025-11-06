import torch
import torch.nn as nn
import torch.nn.functional as F

class EmberMalwareNet(nn.Module):
    """
    Rete neurale per malware detection su dataset EMBER.
    EMBER ha 2381 features estratte da file PE Windows.
    """
    def __init__(self, input_dim=2381, dropout_rate=0.2):
        super(EmberMalwareNet, self).__init__()
        
        # Encoder con BatchNorm e Dropout per regolarizzazione
        self.fc1 = nn.Linear(input_dim, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.dropout1 = nn.Dropout(dropout_rate)
        
        self.fc2 = nn.Linear(1024, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        self.fc3 = nn.Linear(512, 256)
        self.bn3 = nn.BatchNorm1d(256)
        self.dropout3 = nn.Dropout(dropout_rate)
        
        # Residual connection per migliorare il gradiente
        self.fc4 = nn.Linear(256, 128)
        self.bn4 = nn.BatchNorm1d(128)
        self.dropout4 = nn.Dropout(dropout_rate)
        
        self.fc5 = nn.Linear(128, 64)
        self.bn5 = nn.BatchNorm1d(64)
        
        # Output layer per classificazione binaria
        self.output = nn.Linear(64, 1)
        
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
        
        # Layer 4
        x = self.fc4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.dropout4(x)
        
        # Layer 5
        x = self.fc5(x)
        x = self.bn5(x)
        x = F.relu(x)
        
        # Output
        x = self.output(x)
        return x


# Funzione per training
def train_model(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output.squeeze(), target.float())
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pred = (torch.sigmoid(output) > 0.5).float()
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
            output = model(data)
            loss = criterion(output.squeeze(), target.float())
            
            total_loss += loss.item()
            pred = (torch.sigmoid(output) > 0.5).float()
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
    
    model = EmberMalwareNet(input_dim=2381, dropout_rate=0.2).to(device)
    
    # Loss e optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                            factor=0.5, patience=5)
    
    # Training loop (esempio con dati fittizi)
    print(f"Modello creato con {sum(p.numel() for p in model.parameters())} parametri")
    print(f"Device: {device}")
    
    # Carica i tuoi dati EMBER qui
    # train_loader = DataLoader(ember_train_dataset, batch_size=256, shuffle=True)
    # val_loader = DataLoader(ember_val_dataset, batch_size=256)
    
    # for epoch in range(50):
    #     train_loss, train_acc = train_model(model, train_loader, criterion, optimizer, device)
    #     val_loss, val_acc = validate_model(model, val_loader, criterion, device)
    #     scheduler.step(val_loss)
    #     print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
    #           f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")