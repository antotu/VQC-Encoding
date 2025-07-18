from torch import nn
from torch import nn
from VQC import VQC


class Model(nn.Module):
    def __init__(self, Encoding, Reuploading, Hadamard, numLayers, numWires, n_outputs):
        super(Model, self).__init__()
        self.model = VQC(Encoding, Reuploading, Hadamard, numLayers, numWires, n_outputs)
                        
    def forward(self, x):
        return self.model(x)