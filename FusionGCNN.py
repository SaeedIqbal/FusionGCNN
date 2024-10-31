import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from sklearn.utils import class_weight
import numpy as np
import os

# ============================ Dataset and DataLoader ============================
class ECGDataset(Dataset):
    def __init__(self, data_path, transform=None):
        self.data_path = data_path
        self.transform = transform
        self.data, self.labels = self.load_data()

    def load_data(self):
        # Assume data files are numpy arrays with features and labels
        data = []
        labels = []
        for file in os.listdir(self.data_path):
            if file.endswith('.npy'):
                file_data = np.load(os.path.join(self.data_path, file), allow_pickle=True)
                data.append(file_data['signal'])
                labels.append(file_data['label'])
        return np.array(data), np.array(labels)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        if self.transform:
            sample = self.transform(sample)
        return torch.Tensor(sample), torch.tensor(label, dtype=torch.long)


class DataBalancer:
    def __init__(self, dataset):
        self.dataset = dataset

    def get_class_weights(self):
        class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(self.dataset.labels), y=self.dataset.labels)
        return torch.tensor(class_weights, dtype=torch.float)


# ============================ Base Models ============================

class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()

    def forward(self, x):
        raise NotImplementedError("Must be implemented by subclass")

# ============================ SigNet ============================

class SigNet(BaseModel):
    def __init__(self, input_channels=1, output_dim=128):
        super(SigNet, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, output_dim, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(output_dim, output_dim)

    def forward(self, x):
        x = self.relu(self.pool(self.conv1(x)))
        x = self.relu(self.pool(self.conv2(x)))
        x = self.relu(self.pool(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# ============================ DualGCNN ============================

class DualGCNN(BaseModel):
    def __init__(self, gnn, signet):
        super(DualGCNN, self).__init__()
        self.gnn = gnn
        self.signet = signet

    def forward(self, x):
        F_g = self.gnn(x)  # Global feature extraction with GNN
        F_c = self.signet(x)  # Local feature extraction with SigNet
        
        # Interchange features for refinement
        F_g_prime = self.gnn(F_c)
        F_c_prime = self.signet(F_g)
        
        return F_g_prime, F_c_prime

# ============================ FusionGCNN ============================

class FusionGCNN(BaseModel):
    def __init__(self, gnn, signet):
        super(FusionGCNN, self).__init__()
        self.gnn = gnn
        self.signet = signet
        self.fc = nn.Linear(512, 256)
        self.gate = nn.Sigmoid()

    def forward(self, F_g_prime, F_c_prime, x):
        F_g_fusion = self.gnn(F_g_prime)
        F_c_fusion = self.signet(F_c_prime)
        F_g_orig = self.gnn(x)
        F_c_orig = self.signet(x)
        
        # Equalization (normalize features)
        F_g_fusion = (F_g_fusion - F_g_fusion.mean()) / F_g_fusion.std()
        F_c_fusion = (F_c_fusion - F_c_fusion.mean()) / F_c_fusion.std()
        F_g_orig = (F_g_orig - F_g_orig.mean()) / F_g_orig.std()
        F_c_orig = (F_c_orig - F_c_orig.mean()) / F_c_orig.std()

        # Concatenate and apply gating mechanism
        concat_features = torch.cat([F_g_fusion, F_c_fusion, F_g_orig, F_c_orig], dim=1)
        G = self.gate(self.fc(concat_features))
        F_gated = G * concat_features

        return F_gated

# ============================ Classifier ============================

class Classifier(BaseModel):
    def __init__(self, input_dim=256, num_classes=5):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(input_dim, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        logits = self.fc(x)
        probabilities = self.softmax(logits)
        return probabilities

# ============================ Training ============================

class Trainer:
    def __init__(self, model, dataloader, criterion, optimizer, device):
        self.model = model
        self.dataloader = dataloader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        for data, target in self.dataloader:
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(self.dataloader)

# ============================ Main Execution ============================

def main():
    # Define paths, hyperparameters
    data_path = '/home/phd/dataset/MITBIH/'
    batch_size = 32
    learning_rate = 0.001
    num_epochs = 20
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset and DataLoader
    dataset = ECGDataset(data_path, transform=transforms.ToTensor())
    data_balancer = DataBalancer(dataset)
    class_weights = data_balancer.get_class_weights().to(device)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Model components
    signet = SigNet()
    gnn = SigNet()  # Placeholder for a proper GNN model
    dual_gcnn = DualGCNN(gnn, signet)
    fusion_gcnn = FusionGCNN(gnn, signet)
    classifier = Classifier()

    # Full Model
    model = nn.Sequential(dual_gcnn, fusion_gcnn, classifier).to(device)

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Trainer
    trainer = Trainer(model, dataloader, criterion, optimizer, device)

    # Training Loop
    for epoch in range(num_epochs):
        avg_loss = trainer.train_epoch()
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

if __name__ == '__main__':
    main()