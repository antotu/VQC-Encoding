from torch import nn

class Model(nn.Module):
    def __init__(self, Encoding, Reuploading, numLayers, numWires, n_outputs):
        super().__init__()

        if str(Encoding).lower() in {"hamiltonian", "ham", "h"}:
            from VQC_hamiltonian import VQC
            # VQC_hamiltonian signature: (Reuploading, numLayers, numWires, n_outputs, t=1.0)
            self.model = VQC(Reuploading, numLayers, numWires, n_outputs)
        else:
            from VQC import VQC
            # Keep your non-hamiltonian VQC signature (if it exists in your project)
            self.model = VQC(Encoding, Reuploading, numLayers, numWires, n_outputs)

        # IMPORTANT: no Softmax for PauliZ expvals
        self.activation = nn.Identity()

    def forward(self, x):
        return self.activation(self.model(x))
