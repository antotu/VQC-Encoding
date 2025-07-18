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
from sklearn.metrics import confusion_matrix
import time
#numComponents = 3
#numFeatures = range(2, 20)
#f = 4

numLayers = int(sys.argv[1])
Reuploading = True if sys.argv[2]== "True" else False
numWires = 3 #numComponents
n_outputs = 1



deviceGPU = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the diabetes.csv 
df = pd.read_csv("diabetes.csv")
y = df.pop("Outcome")
X = df
X = X.values
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

"""
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


pca = PCA(n_components=numComponents)
X_train = pca.fit_transform(X_train, y_train)
X_test = pca.transform(X_test)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
"""
scaler2 = MinMaxScaler(feature_range=(0, 1))
X_train = scaler2.fit_transform(X_train)
X_test = scaler2.transform(X_test)

#X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, stratify=y_train)
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)
# Rescale y_train from [-1, 1] to [0, 1]

epochs = 30


    
model = Model(Reuploading, numLayers, numWires, n_outputs).to(deviceGPU)

# Define the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
# loss function
criterion = nn.BCELoss()

def evaluation(model, loader):
    # Evaluate the model on the validation set
    model.eval()
    loss_value = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(deviceGPU), labels.to(deviceGPU)
            outputs = (1 - model(inputs)) / 2
            loss = criterion(outputs, labels)
            loss_value += loss.item()
            predicted = (outputs >= 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return loss_value / len(loader), 100 * correct / total


def confusion_matrix_Evaluation(model, inputs, labels):
    # Evaluate the model on the validation set
    model.eval()
    tn, fp, fn, tp = 0, 0, 0, 0
    with torch.no_grad():
        inputs, labels = inputs.to(deviceGPU), labels.to(deviceGPU)
        outputs = (1 - model(inputs)) / 2
        predicted = (outputs >= 0.5).float()
        predicted = predicted.cpu().detach().numpy()
        labels = labels.cpu().detach().numpy()
            
        tn_1, fp_1, fn_1, tp_1 = confusion_matrix(y_pred =predicted, y_true = labels).ravel()
        tn += tn_1
        fp += fp_1
        fn += fn_1
        tp += tp_1
    return tn, fp, fn, tp

def train(X_train, y_train, batch_size=64):
        """
        function used to train the model
        @ X_train: torch tensor of training inputs
        @ y_train: torch tensor of training labels
        @ X_valid: torch tensor of validation inputs
        @ y_valid: torch tensor of validation labels
        """
        
        # Create a DataLoader for the training and validation data
        train_data = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

        

        # Define early stopping criteria
        best_loss = float('inf')
        counter = 0
        best_model_state = None

        # Train the model
        for epoch in range(epochs):
            
            epoch_loss = 0.0
            correct = 0
            total = 0

            # Set the model to training mode
            model.train()

            # Iterate over the mini-batches in the training data
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(deviceGPU), labels.to(deviceGPU)
                optimizer.zero_grad()
                
                # Rescale the output in the interval [0, 1]
                outputs = (1 - model(inputs)) / 2
                loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                
                # If the predicted value is greater than 0.5 (the defined threshold)
                predicted = (outputs >= 0.5).float()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            epoch_loss /= len(train_loader)
            accuracy = 100 * correct / total

            # Print training metrics for the current epoch
            print(f"Epoch {epoch+1}: Train Loss = {epoch_loss:.4f}, Train Accuracy = {accuracy:.2f}%")

start_time_train = time.time()            
train(X_train, y_train,)
end_time_train = time.time()
time_difference_train = end_time_train - start_time_train
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=10, shuffle=False)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=10, shuffle=False)

lossTrain, accuracyTrain = evaluation(model, train_loader)
lossTest, accuracyTest = evaluation(model, test_loader)
print(f"Train Loss = {lossTrain:.4f}, Train Accuracy = {accuracyTrain:.2f}%")
print(f"Test Loss = {lossTest:.4f}, Test Accuracy = {accuracyTest:.2f}%")


def append_dict_to_jsonl(file_name, new_dict):
    """
    Questa funzione legge un file JSON, aggiunge un nuovo dizionario al contenuto,
    e riscrive il file aggiornato.

    Parametri:
    nome_file (str): Il percorso del file JSON da aggiornare.
    nuovo_dizionario (dict): Il nuovo dizionario da aggiungere alla lista di dizionari.
    """
    # 1. Leggere i dati dal file JSON
    try:
        with open(file_name, 'r') as file:
            data = json.load(file)
    except FileNotFoundError:
        # Se il file non esiste, iniziamo con una lista vuota
        data = []
    except json.JSONDecodeError:
        # Se il file è vuoto o contiene dati non validi, iniziamo con una lista vuota
        data = []

    # 2. Controllare se il contenuto è un dizionario o una lista
    if isinstance(data, dict):
        # Se è un dizionario singolo, lo trasformiamo in una lista di dizionari
        data_list = [data]
    elif isinstance(data, list):
        # Se è già una lista, continuiamo a usare quella
        data_list = data
    else:
        raise TypeError("Il formato del JSON non è supportato.")

    # 3. Aggiungere il nuovo dizionario alla lista
    data_list.append(new_dict)

    # 4. Scrivere i dati aggiornati nel file JSON
    with open(file_name, 'w') as file:
        json.dump(data_list, file, indent=4)

    # Conferma del successo (opzionale)
    #print(f"Dati aggiornati salvati nel file: {file_name}")

time_evaluate_train_start = time.time()
tn_train, fp_train, fn_train, tp_train = confusion_matrix_Evaluation(model, torch.tensor(X_train), torch.tensor(y_train))
time_evaluate_train_end = time.time()
tn_test, fp_test, fn_test, tp_test = confusion_matrix_Evaluation(model, torch.tensor(X_test), torch.tensor(y_test))
time_evaluate_test_end = time.time()

time_difference_evaluate_train = time_evaluate_train_end - time_evaluate_train_start
time_difference_evaluate_test = time_evaluate_test_end - time_evaluate_train_end

print(f"Train: TP = {tp_train}, FP = {fp_train}, TN = {tn_train}, FN = {fn_train}")
print(f"Test: TP = {tp_test}, FP = {fp_test}, TN = {tn_test}, FN = {fn_test}")

Encoding="Amplitude"
Hadamard = False
fileNameAcc = f"model_Encoding_{Encoding}_numLayers_{numLayers}_Hadamard_{Hadamard}_Reuploading_{Reuploading}_accuracy.json" 
dictAcc = {"train": accuracyTrain, "test": accuracyTest, "tp_train": tp_train.item(), "fp_train": fp_train.item(), "tn_train": tn_train.item(), "fn_train": fn_train.item(), "tp_test": tp_test.item(), "fp_test": fp_test.item(), "tn_test": tn_test.item(), "fn_test": fn_test.item(), "timeTrain": time_difference_train, "timeEvaluateTrain": time_difference_evaluate_train, "timeEvaluateTest": time_difference_evaluate_test}
append_dict_to_jsonl(fileNameAcc, dictAcc)
fileNameLoss = f"model_Encoding_{Encoding}_numLayers_{numLayers}_Hadamard_{Hadamard}_Reuploading_{Reuploading}_loss.json"
dictLoss = {"train": lossTrain, "test": lossTest}
append_dict_to_jsonl(fileNameLoss, dictLoss)
"""
# Save the accuracy on the 3 datasets in a json
with open(f"model_numFeatures_{f}_numVQC_{numVQC}_numLayers_{numLayers}_accuracy.json", "w") as fileName:
    json.dump({"train": accuracyTrain, "valid": accuracyValid, "test": accuracyTest}, fileName)
# Save the loss on the 3 datasets in a json
with open(f"model_numFeatures_{f}_numVQC_{numVQC}_numLayers_{numLayers}_loss.json", "w") as fileName:
    json.dump({"train": lossTrain, "valid": lossValid, "test": lossTest}, fileName)
# Save the model parameters
#torch.save(model.state_dict(), f"model_numFeatures_{f}_numVQC_{numVQC}_numLayers_{numLayers}_params.pth")

"""