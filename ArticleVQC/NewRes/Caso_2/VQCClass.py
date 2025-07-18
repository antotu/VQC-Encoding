import pennylane as qml
import pennylane.numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score



"""
minibatch_generator(X, y, minibatch_size)
PARAMS
    X input features dataset
    y labels
    minibatch_size size of the minibatch for the optimization
"""

def minibatch_generator(X, y, minibatch_size):
    # get the list of indices of the dataset
    indices = np.arange(X.shape[0])
    # shuffle the list
    np.random.shuffle(indices)
    # create the minibatches
    for start_idx in range(0, indices.shape[0] - minibatch_size + 1, minibatch_size):
        batch_idx = indices[start_idx:start_idx + minibatch_size]
        yield X[batch_idx], y[batch_idx]





class VQC:

    def __init__(self, encoding, solver, numLayer, reUploading, numClasses, numWires, gates=None):
        # encoding parameter can be "amplitude", "angle"
        self.encoding = encoding
        self.gates = gates
        self.solver = solver
        self.numLayer = numLayer
        self.reUploading = reUploading
        self.numClasses = numClasses
        self.num_wires = numWires

        if numClasses > numWires:
            self.num_wires = self.numClasses
            #print("More classes than wires: error")
            #exit(1)

        shape = qml.StronglyEntanglingLayers.shape(n_layers=self.numLayer, n_wires=self.num_wires)
        self.weights = np.pi * np.random.random(size=shape, requires_grad=True)
        #print(self.weights, self.num_wires)
        self.circ = qml.QNode(self.circuit, self.device())
        #print(n_wires, self.num_wires)

    def device(self):
        #print(self.num_wires)
        return qml.device("default.qubit", wires=self.num_wires)

    def superposition(self, num_wires):
        for i in range(num_wires):
            qml.Hadamard(wires=i)

    def set_encoding(self, features):
        #print(self.encoding)
        if self.encoding == "amplitude":
            features = features / np.linalg.norm(features)
            f = np.pad(features, (0, 2 ** self.num_wires - len(features)), "constant", constant_values=0)
            qml.MottonenStatePreparation(state_vector=f, wires=range(self.num_wires))
            #qml.AmplitudeEmbedding(features=features, wires=range(self.num_wires), pad_with=0, normalize=True)
        elif self.encoding == "angle":
            possible_gate = ["X", "XY", "XYZ", "XZ", "XZY", "Y", "YX", "YXZ", "YZ", "YZX",
                             "Y_H", "YX_H", "YXZ_H", "YZ_H", "YZX_H", "Z_H", "ZX_H", "ZXY_H", "ZY_H", "ZYX_H"]
            gate_applied = self.gates
            if self.gates in possible_gate:
                if "_" in self.gates:
                    gate_applied = self.gates.split("_")[0]

                    self.superposition(num_wires=self.num_wires)
                for rot in gate_applied:
                    qml.AngleEmbedding(features=features, wires=range(len(features)), rotation=rot)
            else:
                print("Error in the encoding gates")
                exit(1)
        else:
            print("Error in the definition of the encoding")
            exit(-1)

    def set_num_meas_wire(self):
        if self.numClasses == 2:
            return 1
        else:
            return self.numClasses

    #@qml.qnode(device(n_wires))
    def circuit(self, features):
        # self.set_encoding()
        features = np.array(features, requires_grad=False)
        #print(self.weights.requires_grad)#, features.requires_grad)
        if self.reUploading:
            for i in range(self.weights.shape[0]):
                self.set_encoding(features=features)
                qml.StronglyEntanglingLayers(np.array([self.weights[i]]), wires=range(self.num_wires))
        else:
            self.set_encoding(features=features)
            qml.StronglyEntanglingLayers(self.weights, wires=range(self.num_wires))

        res = []
        for i in range(self.set_num_meas_wire()):
            res.append(qml.probs(wires=[i]))
        return res

    def single_predict(self, features):

        resVal = self.circ(features)
        #print(resVal[0], resVal)

        if self.numClasses == 2:
            res = resVal[0]
            if res[1] >= 0.5:
                return 1
            else:

                return 0
        else:
            res = [x[1] for x in resVal]
            return np.argmax(res)

    def predict(self, X):

        return [self.single_predict(x) for x in X]



    def draw_circuit(self, features):
        qml.draw_mpl(self.circ, fontsize="xx-large", expansion_strategy="device", style="default", decimals=2)(features)
        plt.show()

    def sumDen(self, x):
        retVal = 0
        for i in x:
            retVal += np.exp(i)
        return retVal

    def softmax(self, x):
        den = self.sumDen(x)
        retVal = []
        for i in x:
            retVal.append(np.exp(i) / den)
        return retVal
    
    def cost(self, w, X, y_t):
        self.weights = w
        res = [self.circ(np.array(x, requires_grad=False)) for x in X]
        res = np.array(res)
        if self.numClasses == 2:
            y_pred = np.stack(res[:, 0, 1])
            loss = 0
            for l, p in zip(y_t, y_pred):
                if l == 0:
                    if 1 - p == 0:
                        loss += 10 ** 10
                    else:
                        loss -= np.log(1 - p)
                else:
                    if p == 0:
                        loss += 10 ** 10
                    else:
                        loss -= np.log(p)
            return loss / len(y_t)
        else:
            loss = 0
            y_pred = [r[:, 1] for r in res]
            y_pred = [self.softmax(p) for p in y_pred]
            y_pred = np.stack(y_pred)
            for l, p in zip(y_t, y_pred):
                if p[l] == 0:
                    loss += 10 ** 10
                else:
                    loss += -np.log(p[l])
            return loss / len(y_t)
    """
    getWeights(self)
        return the weights of the VQC
    """
    def getWeights(self):
        return self.weights


    """
    fit(self, X_train, y_train, X_valid, y_valid, optimizer, num_epochs=60, minibatch_size=10)
    PARAMS
        X_train: training set for the optimization
        y_train: labels of the training set
        X_valid: validation set
        y_valid: labels of the validation set
        optimizer: optimizer used for the training
        num_epochs: number of epochs used for the optimization. Default: 30
        minibatch_size: dimension of the minibatch. Default: 10
        
    OUTPUT:
        accuracy list training, the i-th element corresponds to the accuracy of the model at 
                                    the i-th epoch of the training set
        accuracy list validation, the i-th element corresponds to the accuracy of the model at 
                                    the i-th epoch of the validation set
                                    
        accuracy of the model at the end of the training procedure for the training set
        
        accuracy of the model at the end of the training procedure for the validation set
        
    """

    def fit(self, X_train, y_train, X_valid, y_valid, optimizer, num_epochs=30, minibatch_size=10):
        # initialize the two lists
        epoch_train_acc = []
        epoch_valid_acc = []
        # repeat the training procedure num_epochs times
        for epoch in range(num_epochs):
            # generate the minibatch of the dimension minibatch_size
            minibatch_gen = minibatch_generator(X=X_train, y=y_train, minibatch_size=minibatch_size)
            # pass each minibatch to the model for the optimization
            for X_train_mini, y_train_mini in minibatch_gen:
                # reset the gradient accumulation
                optimizer.reset()
                # optimize the parameters
                w, _, _ = optimizer.step(self.cost, self.weights, X_train_mini, y_train_mini)
                # store the new weights
                self.weights = np.array(w, requires_grad=True)

            # predict the outcomes of the training set
            y_pred_train = self.predict(X_train)
            # predict the outcomes of the validation set
            y_pred_valid = self.predict(X_valid)
            # calculate the accuracy of the training set
            acc_train = accuracy_score(y_true=y_train, y_pred=y_pred_train)
            # calculate the accuracy of the validation set
            acc_valid = accuracy_score(y_true=y_valid, y_pred=y_pred_valid)
            # calculate the loss on the training set
            lossTrain = self.cost(self.weights, X_train, y_train)
            # print a report of the results
            print(f"Epoch {epoch + 1:03d}/{num_epochs:03d} | Train Acc: {acc_train * 100:.2f}% |" +
                  f" Valid Acc: {acc_valid * 100:.2f}% | Loss {lossTrain:0.3f}")
            # append the accuracy on the training and test lists
            epoch_train_acc.append(acc_train)
            epoch_valid_acc.append(acc_valid)

        return epoch_train_acc, epoch_valid_acc, epoch_train_acc[-1] ,epoch_valid_acc[-1]

    def getInfo(self, listInfo, features):
        dictRes = {}
        for info in listInfo:

            dictRes[info] = qml.specs(self.circ, expansion_strategy="device")(features)[info]
        return dictRes










