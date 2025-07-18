import pennylane as qml
from CV_VQCClass import CV_VQC
import pandas as pd
import pennylane.numpy as np
import math
from VQCClass import VQC
df = pd.read_csv("diabetes.csv")
y = df.pop("Outcome")
X = df


X = X.values
y = np.array(y)
X_temp = X
y_temp = y

opt = qml.AdamOptimizer(0.01)
cv_vqc = CV_VQC(X_temp, y_temp, opt, ["amplitude", "angle"], ["X", "XY", "XYZ", "XZ", "XZY", "Y", "YX", "YXZ", "YZ", "YZX",
                             "Y_H", "YX_H", "YXZ_H", "YZ_H", "YZX_H", "Z_H", "ZX_H", "ZXY_H", "ZY_H", "ZYX_H"],
                [2, 4, 6, 8, 10], [False, True], 10, 30, 10)
scaler = cv_vqc.preprocessing()
accTrain, accTest = cv_vqc.collectValue(3)
print(accTrain, accTest)
cv_vqc.writeCSV_File(accTrain, "trainDiabetes.csv")
cv_vqc.writeCSV_File(accTest, "testDiabetes.csv")
