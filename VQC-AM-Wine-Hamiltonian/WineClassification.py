import sys
import json
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import confusion_matrix
from sklearn.datasets import load_wine

from model_hamiltonian import Model  # use the fixed Model below

# -------------------------
# CLI: WineClassification.py <numLayers> <Reuploading>
# Example: python3 WineClassification.py 4 True
# -------------------------
if len(sys.argv) < 3:
    raise ValueError("Usage: python3 WineClassification.py <numLayers> <Reuploading(True/False)>")

Encoding = "Hamiltonian"
numLayers = int(sys.argv[1])
Reuploading = True if sys.argv[2] == "True" else False

numComponents = 4
numWires = numComponents
n_outputs = 3
epochs = 30

# PennyLane default.qubit is CPU-based; keep CPU unless you change the PL device backend
deviceGPU = torch.device("cpu")

# Load dataset
X, y = load_wine(return_X_y=True)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=0
)

# Standardize and PCA
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

pca = PCA(n_components=numComponents)
X_train = pca.fit_transform(X_train, y_train)
X_test = pca.transform(X_test)

scaler2 = MinMaxScaler(feature_range=(0, 1))
X_train = scaler2.fit_transform(X_train)
X_test = scaler2.transform(X_test)

# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32, device=deviceGPU)
y_train = torch.tensor(y_train, dtype=torch.long, device=deviceGPU)
X_test = torch.tensor(X_test, dtype=torch.float32, device=deviceGPU)
y_test = torch.tensor(y_test, dtype=torch.long, device=deviceGPU)

model = Model(
    Encoding=Encoding,
    Reuploading=Reuploading,
    numLayers=numLayers,
    numWires=numWires,
    n_outputs=n_outputs,
).to(deviceGPU)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

def evaluation(model, loader):
    model.eval()
    loss_value = 0.0
    correct = 0
    total = 0
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(deviceGPU), labels.to(deviceGPU)
            outputs = model(inputs)

            loss = criterion(outputs, labels)
            loss_value += loss.item()

            predicted = torch.argmax(outputs, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_labels.extend(labels.detach().cpu().numpy().tolist())
            all_predictions.extend(predicted.detach().cpu().numpy().tolist())

    return loss_value / max(len(loader), 1), 100.0 * correct / max(total, 1), all_labels, all_predictions

def train(X_train, y_train, batch_size=64):
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(deviceGPU), labels.to(deviceGPU)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            predicted = torch.argmax(outputs, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss /= max(len(train_loader), 1)
        accuracy = 100.0 * correct / max(total, 1)
        print(f"Epoch {epoch+1}: Train Loss = {epoch_loss:.4f}, Train Accuracy = {accuracy:.2f}%")

def append_dict_to_jsonl(file_name, new_dict):
    try:
        with open(file_name, "r") as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        data = []

    if isinstance(data, dict):
        data = [data]
    elif not isinstance(data, list):
        raise TypeError("Unsupported JSON format")

    data.append(new_dict)

    with open(file_name, "w") as f:
        json.dump(data, f, indent=4)

# Train
t_train0 = time.time()
train(X_train, y_train)
t_train1 = time.time()
training_time = t_train1 - t_train0

# Evaluate
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=10, shuffle=False)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=10, shuffle=False)

t_eval0 = time.time()
lossTrain, accuracyTrain, y_true_train, y_pred_train = evaluation(model, train_loader)
t_eval1 = time.time()
lossTest, accuracyTest, y_true_test, y_pred_test = evaluation(model, test_loader)
t_eval2 = time.time()

cm_train = confusion_matrix(y_true_train, y_pred_train)
cm_test = confusion_matrix(y_true_test, y_pred_test)

fileNameAcc = f"model_Encoding_{Encoding}_numLayers_{numLayers}_Reuploading_{Reuploading}_metrics.json"
metrics = {
    "train": {"accuracy": accuracyTrain, "loss": lossTrain, "confusion_matrix": cm_train.tolist()},
    "test": {"accuracy": accuracyTest, "loss": lossTest, "confusion_matrix": cm_test.tolist()},
    "time": {
        "training_time": training_time,
        "evaluation_train_time": (t_eval1 - t_eval0),
        "evaluation_test_time": (t_eval2 - t_eval1),
    },
}

append_dict_to_jsonl(fileNameAcc, metrics)

print(f"Train Accuracy: {accuracyTrain:.2f}%")
print(f"Test Accuracy: {accuracyTest:.2f}%")
print(f"Train Confusion Matrix:\n{cm_train}")
print(f"Test Confusion Matrix:\n{cm_test}")
