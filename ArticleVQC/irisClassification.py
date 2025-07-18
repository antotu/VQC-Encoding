from sklearn.datasets import load_iris
import pennylane as qml
from VQCClass import VQC
from CV_VQCClass import CV_VQC
X, y = load_iris(return_X_y=True)
X_temp = X
y_temp = y
opt = qml.AdamOptimizer(0.01)
#vqc = VQC(encoding="angle", solver=opt, numLayer=2, reUploading=False, numClasses=3, numWires=X.shape[1], gates="X")
#print(vqc.single_predict(X[0]))
cv_vqc = CV_VQC(X_temp, y_temp, opt, ["angle", "amplitude"], ["X", "XY", "XYZ", "XZ", "XZY", "Y", "YX", "YXZ", "YZ", "YZX",
                             "Y_H", "YX_H", "YXZ_H", "YZ_H", "YZX_H", "Z_H", "ZX_H", "ZXY_H", "ZY_H", "ZYX_H"],
                [2, 4, 6, 8, 10], [False, True], 10, 30, 10)
scaler = cv_vqc.preprocessing()

accTrain, accTest = cv_vqc.collectValue(X_temp.shape[1])
print(accTrain, accTest)
cv_vqc.writeCSV_File(accTrain, "trainIris.csv")
cv_vqc.writeCSV_File(accTest, "testIris.csv")
