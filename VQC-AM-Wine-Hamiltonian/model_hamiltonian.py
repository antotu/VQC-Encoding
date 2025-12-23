from torch import nn

class Model(nn.Module):
    def __init__(self, Encoding, Reuploading, numLayers, numWires, n_outputs):
        super().__init__()

        if str(Encoding).lower() in {"hamiltonian", "ham", "h"}:
            from VQC_hamiltonian import VQC
            self.model = VQC(Reuploading, numLayers, numWires, n_outputs)
        else:
            from VQC import VQC
            self.model = VQC(Encoding, Reuploading, numLayers, numWires, n_outputs)

        # For wine (multi-class), you want logits of shape [batch, n_outputs]
        # Do NOT apply Softmax here; CrossEntropyLoss expects raw logits.
        self.activation = nn.Identity()

    def forward(self, x):
        return self.activation(self.model(x))
