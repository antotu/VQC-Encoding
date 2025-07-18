from VQCClass import VQC
from pennylane import numpy as np
import math
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from sklearn.decomposition import PCA
class CV_VQC:
    def __init__(self, X, y, optimizer, encoding, gateAngle=["angle"], numParamLayer=["Y"],
                 reUploading=[False], cv=10, num_epochs=30, minibatch_size=5):

        self.X_train, self.X_valid, self.y_train, self.y_valid = train_test_split(X, y, test_size=0.2, stratify=y)
        self.optimizer = optimizer
        self.encoding = encoding
        self.paramAngle = gateAngle
        self.numParamLayer = numParamLayer
        self.cv = cv
        self.reUploading = reUploading
        self.num_epochs = num_epochs
        self.numClasses = len(np.unique(self.y_train))
        self.minibatch_size = minibatch_size

    def preprocessing(self):
        scaler = MinMaxScaler(feature_range=(0, math.pi))
        self.X_train = scaler.fit_transform(self.X_train)
        self.X_valid = scaler.transform(self.X_valid)
        return scaler

    def reductionFeatures(self, redFeatures=False, numComponents=4):
        if redFeatures:
            pca = PCA(n_components=numComponents)
            X_train = pca.fit_transform(self.X_train, self.y_train)
            X_valid = pca.transform(self.X_valid)
        else:
            X_train = self.X_train
            X_valid = self.X_valid
        return X_train, X_valid


    def collectValue(self, numMaxWires):
        retValTrain = {}
        retValTest = {}
        for e in self.encoding:
            if e == "angle":
                for g in self.paramAngle:
                    for p in self.numParamLayer:
                        for r in self.reUploading:
                            circAn = e + "_" + g + "_" + str(p) + "_" + str(r)
                            print(circAn)
                            #meanTrain = np.zeros(self.num_epochs,)
                            #meanValid = np.zeros(self.num_epochs, )
                            #print(circAn)
                            trainVal = []
                            validVal = []

                            for rep in range(self.cv):
                                if numMaxWires < self.X_train.shape[1]:
                                    X_train, X_valid = self.reductionFeatures(True, numMaxWires)
                                else:
                                    X_train, X_valid = self.reductionFeatures(False)
                                vqc = VQC(encoding=e, gates=g, solver=self.optimizer, numLayer=p, reUploading=r, numClasses=self.numClasses, numWires=X_train.shape[1])
                                #vqc.draw_circuit(self.X_train[0])

                                _, _, epoch_train_acc, epoch_valid_acc = vqc.fit(X_train, self.y_train, X_valid, self.y_valid,
                                                                           optimizer=self.optimizer, num_epochs=self.num_epochs,
                                                                           minibatch_size=self.minibatch_size)

                                trainVal.append(epoch_train_acc)
                                validVal.append(epoch_valid_acc)

                            meanTrain, stdTrain, maxTrain = np.array(trainVal).mean(), np.array(trainVal).std(), np.array(trainVal).max()
                            meanValid, stdValid, maxValid = np.array(validVal).mean(), np.array(validVal).std(), np.array(validVal).max()

                            retValTrain[circAn] = (meanTrain, stdTrain, maxTrain)
                            retValTest[circAn] = (meanValid, stdValid, maxValid)
            else:
                for p in self.numParamLayer:
                    #for r in self.reUploading:
                    circAn = e + "_" + str(p)

                    print(circAn)
                    trainVal = []
                    validVal = []
                    for rep in range(self.cv):
                        if numMaxWires < math.ceil(math.log(self.X_train.shape[1], 2)):
                            X_train, X_valid = self.reductionFeatures(True, 2 ** numMaxWires)
                        else:
                            X_train, X_valid = self.reductionFeatures(False)
                        vqc = VQC(encoding=e, solver=self.optimizer, numLayer=p, reUploading=False, numClasses=self.numClasses,
                                  numWires=math.ceil(math.log(self.X_train.shape[1], 2)))

                        _, _, epoch_train_acc, epoch_valid_acc = vqc.fit(X_train, self.y_train, X_valid,
                                                                    self.y_valid,
                                                                    optimizer=self.optimizer, num_epochs=self.num_epochs,
                                                                    minibatch_size=self.minibatch_size)

                        trainVal.append(epoch_train_acc)
                        validVal.append(epoch_valid_acc)


                    meanTrain, stdTrain, maxTrain = np.array(trainVal).mean(), np.array(trainVal).std(), np.array(trainVal).max()
                    meanValid, stdValid, maxValid = np.array(validVal).mean(), np.array(validVal).std(), np.array(validVal).max()

                    retValTrain[circAn] = (meanTrain, stdTrain, maxTrain)
                    retValTest[circAn] = (meanValid, stdValid, maxValid)
        return retValTrain, retValTest



    def writeCSV_File(self, dictVal, name):
        df = pd.DataFrame(dictVal)
        resDF = df.T
        resDF = resDF.set_axis(["mean", "std", "max"], axis = 1, inplace = False)
        resDF.to_csv(name)


    