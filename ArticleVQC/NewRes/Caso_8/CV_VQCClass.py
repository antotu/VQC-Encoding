from VQCClass import VQC
from pennylane import numpy as np
import math
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from sklearn.decomposition import PCA


class CV_VQC:
    """
    CONSTRUCTOR
    INPUTS:
        1) X: features dataset
        2) y: associated labels
        3) optimizer: optimizer used for the classification
        4) encoding: list of string to choose which type of encoding is chosen.
            The possibilities are
                a) angle
                b) amplitude
            Default ["angle"]
        5) numParamLayer: list of integer used to parametrize the number of layer of the parametric circuit. Default [4]
        6) reUploading: list of boolean, if the reUploading is applied for the corresponding model True, otherwise False.
                        Default [False]
        7) cv: integer, sets the number of times the models are tested. Default 10
        8) num_epochs: integer, number of time the entire dataset is passed to the VQC for the optimization. Default 30
        9) minibatch_size: integer, dimension of the minibatch for the optimization part. Default: 5
    """

    def __init__(self, X, y, optimizer, encoding, gateAngle=["Y"], numParamLayer=[4],
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

    """
        preprocessing(self)
        This method is used for preprocess the dataset using MinMax scaler
        For the training dataset, it finds the max and min value and scales as follows
        (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
        The test set is transformed using the same min and max values for the training set
    """
    def preprocessing(self):
        scaler = MinMaxScaler(feature_range=(0, math.pi))
        self.X_train = scaler.fit_transform(self.X_train)
        self.X_valid = scaler.transform(self.X_valid)
        return scaler

    """
    reductionFeatures(self, redFeatures=False, numComponents=4)
    PARAMS:
        1) redFeatures. Boolean, if False no transformation is applied, 
            otherwise the transformation is applied. Deafult: False
        2) numComponents. Integer, number of components desired
        
    If redFeatures is True, PCA is applied to reduce the number of features  
    """

    def reductionFeatures(self, redFeatures=False, numComponents=4):
        if redFeatures:
            pca = PCA(n_components=numComponents)
            X_train = pca.fit_transform(self.X_train, self.y_train)
            X_valid = pca.transform(self.X_valid)
        else:
            X_train = self.X_train
            X_valid = self.X_valid
        return X_train, X_valid

    """
    collectValue(self, numMaxWires)
    PARAMS
        numMaxWires: integer, set the number max of wires of the model to develop
        
        This function is used to check the accuracies on the training and test set
        of all the models that can be developed
    """

    def collectValue(self, numMaxWires, nameDict):
        # init the two dictionary
        retValTrain = {}
        retValTest = {}

        # check all the possible combination
        for e in self.encoding:
            # check if the encoding approach is the angle one
            if e == "angle":
                # loop for each combination of rotational gates
                for g in self.paramAngle:
                    # loop for each combination of number of parameters layer
                    for p in self.numParamLayer:
                        # loop for each strategy of reuploading
                        for r in self.reUploading:
                            # initialize string name of the model and the
                            # two lists that stores the result on the training and test set
                            circAn = e + "_" + g + "_" + str(p) + "_" + str(r)
                            trainVal = []
                            validVal = []
                            # repeat the training and the test procedure for cv times
                            for rep in range(self.cv):
                                print(circAn + f"\tRep: {rep}")
                                # reduce the number of features if it is necessary
                                if numMaxWires < self.X_train.shape[1]:
                                    X_train, X_valid = self.reductionFeatures(True, numMaxWires)
                                else:
                                    X_train, X_valid = self.reductionFeatures(False)
                                # initialize the VQC model
                                vqc = VQC(encoding=e, gates=g, solver=self.optimizer, numLayer=p, reUploading=r, numClasses=self.numClasses, numWires=X_train.shape[1])

                                # train the VQC model and obtain the results of the accuracy and validation
                                _, _, epoch_train_acc, epoch_valid_acc = vqc.fit(X_train,
                                                                                 self.y_train,
                                                                                 X_valid,
                                                                                 self.y_valid,
                                                                                 optimizer=self.optimizer,
                                                                                 num_epochs=self.num_epochs,
                                                                                 minibatch_size=self.minibatch_size)
                                # store the results on the lists
                                trainVal.append(epoch_train_acc)
                                validVal.append(epoch_valid_acc)

                            # calculate the mean, the standard deviation and find the max value of the accuracy list for the training and test
                            meanTrain, stdTrain, maxTrain = np.array(trainVal).mean(), np.array(trainVal).std(), np.array(trainVal).max()
                            meanValid, stdValid, maxValid = np.array(validVal).mean(), np.array(validVal).std(), np.array(validVal).max()

                            dictTmpTrain = {circAn: (meanTrain, stdTrain, maxTrain)}
                            dictTmpTest = {circAn: (meanValid, stdValid, maxValid)}
                            self.writeCSV_File(dictTmpTrain, "train" + nameDict + ".csv", "a")
                            self.writeCSV_File(dictTmpTest, "test" + nameDict + ".csv", "a")

                            # store the result on the dictionary
                            retValTrain[circAn] = (meanTrain, stdTrain, maxTrain)
                            retValTest[circAn] = (meanValid, stdValid, maxValid)
            else:

                # loop for each combination of number of parameters layer
                for p in self.numParamLayer:
                    # initialize string name of the model and the
                    # two lists that stores the result on the training and test set

                    for r in self.reUploading:
                        # repeat the training and the test procedure for cv times
                        circAn = e + "_" + str(p) + "_" + str(r)
                        print(circAn)
                        trainVal = []
                        validVal = []
                        for rep in range(self.cv):
                            # reduce the number of features if it is necessary
                            if numMaxWires < math.ceil(math.log(self.X_train.shape[1], 2)):
                                X_train, X_valid = self.reductionFeatures(True, 2 ** numMaxWires)
                            else:
                                X_train, X_valid = self.reductionFeatures(False)
                            # initialize the VQC model
                            vqc = VQC(encoding=e, solver=self.optimizer, numLayer=p, reUploading=r,
                                      numClasses=self.numClasses,
                                      numWires=math.ceil(math.log(self.X_train.shape[1], 2)))
                            # train the VQC model and obtain the results of the accuracy and validation
                            _, _, epoch_train_acc, epoch_valid_acc = vqc.fit(X_train, self.y_train, X_valid,
                                                                        self.y_valid,
                                                                        optimizer=self.optimizer, num_epochs=self.num_epochs,
                                                                        minibatch_size=self.minibatch_size)
                            # store the results on the lists
                            trainVal.append(epoch_train_acc)
                            validVal.append(epoch_valid_acc)

                        # calculate the mean, the standard deviation and find the max value of the accuracy list for the
                        # training and test
                        meanTrain, stdTrain, maxTrain = np.array(trainVal).mean(), np.array(trainVal).std(),\
                                                        np.array(trainVal).max()
                        meanValid, stdValid, maxValid = np.array(validVal).mean(), np.array(validVal).std(),\
                                                        np.array(validVal).max()


                        dictTmpTrain = {circAn: (meanTrain, stdTrain, maxTrain)}
                        dictTmpTest = {circAn: (meanValid, stdValid, maxValid)}
                        self.writeCSV_File(dictTmpTrain, "train" + nameDict + ".csv", "a")
                        self.writeCSV_File(dictTmpTest, "test" + nameDict + ".csv", "a")
                        # store the result on the dictionary
                        retValTrain[circAn] = (meanTrain, stdTrain, maxTrain)
                        retValTest[circAn] = (meanValid, stdValid, maxValid)

        # return the dictionaries
        return retValTrain, retValTest

    """
    writeCSV_File(self, dictVal, name)
    PARAMS
        dictVal: dictionary containing the information to write into the file
        name: string. name of the file
    """

    """def collectDepth(self):
        # init the two dictionary
        retValTrain = {}
        retValTest = {}

        # check all the possible combination
        for e in self.encoding:
            # check if the encoding approach is the angle one
            if e == "angle":
                # loop for each combination of rotational gates
                for g in self.paramAngle:
                    # loop for each combination of number of parameters layer
                    for p in self.numParamLayer:
                        # loop for each strategy of reuploading
                        for r in self.reUploading:
                            vqc = VQC(encoding=e, gates=g, solver=self.optimizer, numLayer=p, reUploading=r,
                                      numClasses=self.numClasses, numWires=X_train.shape[1])"""

    def writeCSV_File(self, dictVal, name, mode="a"):
        df = pd.DataFrame(dictVal)
        resDF = df.T
        resDF = resDF.set_axis(["mean", "std", "max"], axis = 1, inplace = False)
        resDF.to_csv(name, mode=mode, header=False)

    
