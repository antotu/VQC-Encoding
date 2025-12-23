from torch import nn

class Model(nn.Module):
    def __init__(self, Encoding, Reuploading, Hadamard, numLayers, numWires, n_outputs):
        super(Model, self).__init__()

        # Select implementation based on Encoding string
        if str(Encoding).lower() in {"hamiltonian", "ham", "h"}:
            from VQC_hamiltonian import VQC
        else:
            from VQC import VQC

        self.model = VQC(Encoding, Reuploading, Hadamard, numLayers, numWires, n_outputs)
        self.activation = nn.Softmax(dim=1)

    def forward(self, x):
        return self.activation(self.model(x))
