import pennylane as qml
import sys
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from VQC import VQC
from model import Model
import sys
import json
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import time
from sklearn.datasets import load_wine

numComponents = 4
X, y = load_wine(return_X_y=True)

Encoding = "Hamiltonian"
numLayers = int(sys.argv[2])
Hadamard = True if sys.argv[3] == "True" else False
Reuploading = True if sys.argv[4]== "True" else False
numWires = numComponents
n_outputs = 3

deviceGPU = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

# Standardize and PCA
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

pca = PCA(n_components=numComponents)
X_train = pca.fit_transform(X_train, y_train)
X_test = pca.transform(X_test)

scaler = MinMaxScaler(feature_range=(0, 1))
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)  # Cambiamo a long per CrossEntropyLoss
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.long)

epochs = 30
model = Model(Encoding, Reuploading, Hadamard, numLayers, numWires, n_outputs).to(deviceGPU)

# Define the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
# Use CrossEntropyLoss for multi-class classification
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

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    return loss_value / len(loader), 100 * correct / total, all_labels, all_predictions

def confusion_matrix_evaluation(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    return cm

def train(X_train, y_train, batch_size=64):
    train_data = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        epoch_loss = 0.0
        correct = 0
        total = 0
        model.train()

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(deviceGPU), labels.to(deviceGPU)
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss /= len(train_loader)
        accuracy = 100 * correct / total
        print(f"Epoch {epoch+1}: Train Loss = {epoch_loss:.4f}, Train Accuracy = {accuracy:.2f}%")

start_time_train = time.time()
train(X_train, y_train)
end_time_train = time.time()
time_difference_train = end_time_train - start_time_train

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=10, shuffle=False)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=10, shuffle=False)

# Evaluation
start_time_eval_train = time.time()
lossTrain, accuracyTrain, y_true_train, y_pred_train = evaluation(model, train_loader)
end_time_eval_train = time.time()
time_difference_eval_train = end_time_eval_train - start_time_eval_train
start_time_test = time.time()
lossTest, accuracyTest, y_true_test, y_pred_test = evaluation(model, test_loader)
end_time_eval_test = time.time()
time_difference_eval_test = end_time_eval_test - start_time_test
# Confusion matrix
cm_train = confusion_matrix_evaluation(y_true_train, y_pred_train)
cm_test = confusion_matrix_evaluation(y_true_test, y_pred_test)

# Log performance and confusion matrix
fileNameAcc = f"model_Encoding_{Encoding}_numLayers_{numLayers}_Hadamard_{Hadamard}_Reuploading_{Reuploading}_metrics.json"
metrics = {
    "train": {
        "accuracy": accuracyTrain,
        "loss": lossTrain,
        "confusion_matrix": cm_train.tolist()  # Convert numpy array to list for JSON
    },
    "test": {
        "accuracy": accuracyTest,
        "loss": lossTest,
        "confusion_matrix": cm_test.tolist()  # Convert numpy array to list for JSON
    },
    "time": {"training_time": time_difference_train,
             "evaluation_train_time": time_difference_eval_train,
             "evaluation_test_time": time_difference_eval_test}
}

# Save metrics and confusion matrix to JSON
def append_dict_to_jsonl(file_name, new_dict):
    try:
        with open(file_name, 'r') as file:
            data = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        data = []

    data.append(new_dict)

    with open(file_name, 'w') as file:
        json.dump(data, file, indent=4)

append_dict_to_jsonl(fileNameAcc, metrics)

# Print results
print(f"Train Accuracy: {accuracyTrain:.2f}%")
print(f"Test Accuracy: {accuracyTest:.2f}%")
print(f"Train Confusion Matrix:\n {cm_train}")
print(f"Test Confusion Matrix:\n {cm_test}")
