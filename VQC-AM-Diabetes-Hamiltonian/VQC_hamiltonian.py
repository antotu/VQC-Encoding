import torch
from torch import nn
import pennylane as qml


class VQC(nn.Module):
    """
    Variational Quantum Classifier (PyTorch + PennyLane)

    This version replaces the previous state-preparation embedding
    (MottonenStatePreparation / amplitude-style) with a *Hamiltonian encoding*:

        U_emb(x; θ) = exp(-i t H(x; θ))

    We implement a simple *commuting* Pauli-sum Hamiltonian (Z and ZZ terms),
    so the embedding is exact (no Trotterization required) and efficient.

    Hamiltonian per layer ℓ:
        H_ℓ(x; θ) = Σ_i ( ω_{ℓ,i} * x_i + b_{ℓ,i} ) Z_i  +  Σ_i γ_{ℓ,i} Z_i Z_{i+1}

    where (ω, b, γ) are trainable, and (i+1) is modulo numWires (ring coupling).
    """

    def __init__(self, Reuploading: bool, numLayers: int, numWires: int, n_outputs: int, t: float = 1.0):
        super().__init__()

        self.Reuploading = Reuploading
        self.numLayers = numLayers
        self.numWires = numWires
        self.n_outputs = n_outputs
        self.t = float(t)

        # PennyLane device
        self.dev = qml.device("default.qubit", wires=self.numWires)

        # Variational (ansatz) parameters: StronglyEntanglingLayers
        self.ansatz_weight_shapes = {"weights": (self.numLayers, self.numWires, 3)}

        # Hamiltonian-embedding parameters per layer and qubit:
        #   [:, :, 0] = omega, [:, :, 1] = bias, [:, :, 2] = gamma (ZZ coupling to next qubit)
        self.embed_weight_shapes = {"embed_weights": (self.numLayers, self.numWires, 3)}

        # Combine for TorchLayer
        self.weight_shapes = {**self.ansatz_weight_shapes, **self.embed_weight_shapes}

        # Set device for PyTorch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)

        # Create the quantum node & TorchLayer
        self.qnode = self.create_qnode()
        self.qlayer = qml.qnn.TorchLayer(self.qnode, self.weight_shapes).to(self.device)

    def create_qnode(self):
        """Creates the quantum node for the hybrid model."""
        @qml.qnode(self.dev, interface="torch")
        def qnode(inputs, weights, embed_weights):
            for i in range(self.numWires):
                qml.Hadamard(wires=i)
            # Encoding + ansatz logic
            if self.Reuploading:
                # Interleave embedding and ansatz per layer
                for l in range(self.numLayers):
                    self.hamiltonian_encoding(inputs, embed_weights[l])
                    self.apply_ansatz(weights[l].unsqueeze(0))
            else:
                # One embedding (use first layer's embedding params), then full ansatz
                self.hamiltonian_encoding(inputs, embed_weights[0])
                self.apply_ansatz(weights)

            # Measurement
            return [qml.expval(qml.PauliZ(wires=i)) for i in range(self.n_outputs)]

        return qnode

    def apply_ansatz(self, weights):
        """Apply the variational circuit (ansatz)."""
        qml.StronglyEntanglingLayers(weights, wires=range(self.numWires))

    def _prepare_features(self, inputs):
        x = inputs

        if x.dim() == 2:
            # batch mode: pad/slice features dimension
            if x.shape[1] > self.numWires:
                x = x[:, : self.numWires]
            elif x.shape[1] < self.numWires:
                pad = torch.zeros(x.shape[0], self.numWires - x.shape[1], dtype=x.dtype, device=x.device)
                x = torch.cat([x, pad], dim=1)
            return x

        # single sample mode
        if x.shape[0] > self.numWires:
            x = x[: self.numWires]
        elif x.shape[0] < self.numWires:
            pad = torch.zeros(self.numWires - x.shape[0], dtype=x.dtype, device=x.device)
            x = torch.cat([x, pad], dim=0)
        return x


    def hamiltonian_encoding(self, inputs, layer_embed_weights):
        """
        Hamiltonian encoding layer:
            U = exp(-i t H(x; θ))
        with commuting terms Z_i and Z_i Z_{i+1}.
        """
        x = self._prepare_features(inputs)

        omega = layer_embed_weights[:, 0]
        bias = layer_embed_weights[:, 1]
        gamma = layer_embed_weights[:, 2]

        # Single-qubit Z terms: exp(-i t (omega*x + bias) Z) == RZ(2 t (omega*x + bias))
        for i in range(self.numWires):
            coeff = omega[i] * x[i] + bias[i]
            qml.RZ(2.0 * self.t * coeff, wires=i)

        # Nearest-neighbor ZZ ring couplings:
        # exp(-i t gamma_i Z_i Z_{i+1}) == MultiRZ(2 t gamma_i) on wires [i, i+1]
        for i in range(self.numWires):
            j = (i + 1) % self.numWires
            qml.MultiRZ(2.0 * self.t * gamma[i], wires=[i, j])

    def forward(self, inputs):
        """Forward pass through the hybrid model."""
        # inputs can be [batch, features] or [features]
        if inputs.dim() == 2:
            # Evaluate one sample at a time
            outputs = [self.qlayer(x) for x in inputs]   # each x is shape [features]
            return torch.stack(outputs, dim=0)           # shape [batch, n_outputs]
        return self.qlayer(inputs)

