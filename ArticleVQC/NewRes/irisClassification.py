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
cv_vqc = CV_VQC(X_temp, y_temp, opt, ["amplitude"], [],
                [2, 4, 6, 8, 10], [True], 10, 30, 10)
scaler = cv_vqc.preprocessing()

accTrain, accTest = cv_vqc.collectValue(X_temp.shape[1], "Iris")
print(accTrain, accTest)
#cv_vqc.writeCSV_File(accTrain, "trainIris.csv")
#cv_vqc.writeCSV_File(accTest, "testIris.csv")
