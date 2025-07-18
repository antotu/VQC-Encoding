import pennylane as qml
import math
import pennylane.numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics
from sklearn.metrics import accuracy_score





def minibatch_generator(X, y, minibatch_size):
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
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
            qml.AmplitudeEmbedding(features=features, wires=range(self.num_wires), pad_with=0, normalize=True)
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
        qml.draw_mpl(self.circ, decimals=2, fontsize="xx-large", expansion_strategy="device")(features)
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
        #print(X[0, :])
        #print(y_pred)

        #print(y_pred)
        if self.numClasses == 2:
            y_pred = np.stack(res[:, 0, 1])
            #print(y_t.shape, y_pred.shape)
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

            #return sklearn.metrics.log_loss(y_true=y_t, y_pred=y_pred)
        else:
            loss = 0
            y_pred = [r[:, 1] for r in res]
            # print(y_pred)
            y_pred = [self.softmax(p) for p in y_pred]
            # print(y_pred)
            y_pred = np.stack(y_pred)
            for l, p in zip(y_t, y_pred):
                #print(p[l], p)
                #print(l, p)
                if p[l] == 0:
                    loss += 10 ** 10
                else:
                    loss += -np.log(p[l])
            #print(np.mean(loss))
            return loss / len(y_t)

    def getWeights(self):
        return self.weights


    def fit(self, X_train, y_train, X_valid, y_valid, optimizer, num_epochs=60, minibatch_size=10):
        epoch_train_acc = []
        epoch_valid_acc = []
        for epoch in range(num_epochs):
            minibatch_gen = minibatch_generator(X=X_train, y=y_train, minibatch_size=minibatch_size)
            for X_train_mini, y_train_mini in minibatch_gen:
                optimizer.reset()
                w, _, _ = optimizer.step(self.cost, self.weights, X_train_mini, y_train_mini)
                self.weights = np.array(w, requires_grad=True)


            y_pred_train = self.predict(X_train)
            y_pred_valid = self.predict(X_valid)
            acc_train = accuracy_score(y_true=y_train, y_pred=y_pred_train)
            acc_valid = accuracy_score(y_true=y_valid, y_pred=y_pred_valid)
            lossTrain = self.cost(self.weights, X_train, y_train)
            print(f"Epoch {epoch + 1:03d}/{num_epochs:03d} | Train Acc: {acc_train * 100:.2f}% | Valid Acc: {acc_valid * 100:.2f}% | Loss {lossTrain:0.3f}")
            epoch_train_acc.append(acc_train)
            epoch_valid_acc.append(acc_valid)
        return epoch_train_acc, epoch_valid_acc, epoch_train_acc[-1] ,epoch_valid_acc[-1]












