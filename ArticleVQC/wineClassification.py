

from sklearn.datasets import load_wine

import pennylane as qml

from CV_VQCClass import CV_VQC
X, y = load_wine(return_X_y=True)
X_temp = X
y_temp = y
opt = qml.AdamOptimizer(0.01)

cv_vqc = CV_VQC(X_temp, y_temp, opt, ["angle", "amplitude"], ["X", "XY", "XYZ", "XZ", "XZY", "Y", "YX", "YXZ", "YZ", "YZX",
                             "Y_H", "YX_H", "YXZ_H", "YZ_H", "YZX_H", "Z_H", "ZX_H", "ZXY_H", "ZY_H", "ZYX_H"],
                [2, 4, 6, 8, 10], [False, True], 10, 30, 10)
scaler = cv_vqc.preprocessing()

accTrain, accTest = cv_vqc.collectValue(4)
print(accTrain, accTest)
cv_vqc.writeCSV_File(accTrain, "trainWine.csv")
cv_vqc.writeCSV_File(accTest, "testWine.csv")
