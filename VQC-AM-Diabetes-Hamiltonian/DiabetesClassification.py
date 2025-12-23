import sys
import json
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import confusion_matrix

from model_hamiltonian import Model  # use the fixed model file below

numComponents = 3
Encoding = "Hamiltonian"

# -------------------------
# CLI: DiabetesClassification.py <numLayers> <Reuploading>
# Example: python3 DiabetesClassification.py 4 True
# -------------------------
if len(sys.argv) < 3:
    raise ValueError("Usage: python3 DiabetesClassification.py <numLayers> <Reuploading(True/False)>")

numLayers = int(sys.argv[1])
Reuploading = True if sys.argv[2] == "True" else False

numWires = numComponents
n_outputs = 1

# PennyLane default.qubit is CPU-based. Keep everything on CPU unless you change device backend.
deviceGPU = torch.device("cpu")

# Load data
df = pd.read_csv("diabetes.csv")
y = df.pop("Outcome").to_numpy()
X = df.to_numpy()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=0
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

pca = PCA(n_components=numComponents)
X_train = pca.fit_transform(X_train, y_train)
X_test = pca.transform(X_test)

scaler2 = MinMaxScaler(feature_range=(0, 1))
X_train = scaler2.fit_transform(X_train)
X_test = scaler2.transform(X_test)

X_train = torch.tensor(X_train, dtype=torch.float32, device=deviceGPU)
y_train = torch.tensor(y_train, dtype=torch.float32, device=deviceGPU).view(-1, 1)
X_test = torch.tensor(X_test, dtype=torch.float32, device=deviceGPU)
y_test = torch.tensor(y_test, dtype=torch.float32, device=deviceGPU).view(-1, 1)

epochs = 30

model = Model(Encoding=Encoding, Reuploading=Reuploading, numLayers=numLayers, numWires=numWires, n_outputs=n_outputs).to(deviceGPU)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.BCELoss()

def evaluation(model, loader):
    model.eval()
    loss_value = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(deviceGPU), labels.to(deviceGPU)

            # model returns expval in [-1, 1]; map to [0, 1]
            outputs = (1.0 - model(inputs)) / 2.0

            loss = criterion(outputs, labels)
            loss_value += loss.item()

            predicted = (outputs >= 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return loss_value / max(len(loader), 1), 100.0 * correct / max(total, 1)

def confusion_matrix_evaluation(model, inputs, labels):
    model.eval()
    with torch.no_grad():
        inputs = inputs.to(deviceGPU)
        labels = labels.to(deviceGPU)

        outputs = (1.0 - model(inputs)) / 2.0
        predicted = (outputs >= 0.5).float()

        y_pred = predicted.detach().cpu().numpy().ravel()
        y_true = labels.detach().cpu().numpy().ravel()

        tn, fp, fn, tp = confusion_matrix(y_true=y_true, y_pred=y_pred).ravel()

    print(tn, fp, fn, tp)
    return int(tn), int(fp), int(fn), int(tp)

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

            outputs = (1.0 - model(inputs)) / 2.0
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            predicted = (outputs >= 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss /= max(len(train_loader), 1)
        accuracy = 100.0 * correct / max(total, 1)
        print(f"Epoch {epoch+1}: Train Loss = {epoch_loss:.4f}, Train Accuracy = {accuracy:.2f}%")

start_time_train = time.time()
train(X_train, y_train)
time_difference_train = time.time() - start_time_train

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=10, shuffle=False)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=10, shuffle=False)

lossTrain, accuracyTrain = evaluation(model, train_loader)
lossTest, accuracyTest = evaluation(model, test_loader)
print(f"Train Loss = {lossTrain:.4f}, Train Accuracy = {accuracyTrain:.2f}%")
print(f"Test Loss = {lossTest:.4f}, Test Accuracy = {accuracyTest:.2f}%")

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

t0 = time.time()
tn_train, fp_train, fn_train, tp_train = confusion_matrix_evaluation(model, X_train, y_train)
t1 = time.time()
tn_test, fp_test, fn_test, tp_test = confusion_matrix_evaluation(model, X_test, y_test)
t2 = time.time()

time_difference_evaluate_train = t1 - t0
time_difference_evaluate_test = t2 - t1

print(f"Train: TP = {tp_train}, FP = {fp_train}, TN = {tn_train}, FN = {fn_train}")
print(f"Test: TP = {tp_test}, FP = {fp_test}, TN = {tn_test}, FN = {fn_test}")

fileNameAcc = f"model_Encoding_{Encoding}_numLayers_{numLayers}_Reuploading_{Reuploading}_accuracy.json"
dictAcc = {
    "train": accuracyTrain,
    "test": accuracyTest,
    "tp_train": tp_train,
    "fp_train": fp_train,
    "tn_train": tn_train,
    "fn_train": fn_train,
    "tp_test": tp_test,
    "fp_test": fp_test,
    "tn_test": tn_test,
    "fn_test": fn_test,
    "timeTrain": time_difference_train,
    "timeEvaluateTrain": time_difference_evaluate_train,
    "timeEvaluateTest": time_difference_evaluate_test,
}
append_dict_to_jsonl(fileNameAcc, dictAcc)

fileNameLoss = f"model_Encoding_{Encoding}_numLayers_{numLayers}_Reuploading_{Reuploading}_loss.json"
dictLoss = {"train": lossTrain, "test": lossTest}
append_dict_to_jsonl(fileNameLoss, dictLoss)
