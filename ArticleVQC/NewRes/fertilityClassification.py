import pennylane as qml
from CV_VQCClass import CV_VQC
import pandas as pd
import pennylane.numpy as np
import math
from VQCClass import VQC
df = pd.read_csv("fertility_Diagnosis.txt", header=None)
y = df.pop(9)
X = df

y = [0 if x == "N" else 1 for x in y]
X[0] = (X[0] /2 + 0.5)
X[5] = (X[5] / 2 + 0.5)
X = X.values
y = np.array(y)
X_temp = X
y_temp = y

opt = qml.AdamOptimizer(0.01)
cv_vqc = CV_VQC(X_temp, y_temp, opt, ["amplitude"], [],
                [2, 4, 6, 8, 10], [True], 10, 30, 10)
scaler = cv_vqc.preprocessing()
accTrain, accTest = cv_vqc.collectValue(4, "Fertility")
print(accTrain, accTest)
#cv_vqc.writeCSV_File(accTrain, "trainFertility.csv")
#cv_vqc.writeCSV_File(accTest, "testFertility.csv")
