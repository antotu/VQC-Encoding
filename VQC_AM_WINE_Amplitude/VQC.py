import torch
from torch import nn
import pennylane as qml
import math
import numpy as np
deviceGPU = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class VQC(nn.Module):
    def __init__(self, Reuploading, numLayers, numWires, n_outputs):

        super().__init__()
        """
        Constructor
        @Encoding: String which represents the gates used for the Angle Encoding
        @Ansatz: String which represents the ansatz used for quantum circuit
        @Reuploading: Boolean indicating whether or not to use reuploading
        @Hadamard: Boolean indicating whether or not to use Hadamard gates
        @numLayers: Integer representing the number of layers in the quantum circuit
        @numWires: Integer representing the number of wires in the quantum circuit 
        """
        self.Reuploading = Reuploading
        self.numLayers = numLayers
        self.numWires = numWires
        self.n_outputs = n_outputs

        # PennyLane device
        self.dev = qml.device("default.qubit", wires=self.numWires)
        """
        # Validate Encoding
        valid_encodings = {'X', 'Y', 'Z'}
        for letter in Encoding:
            if letter not in valid_encodings:
                raise ValueError(f"Invalid encoding gate: {letter}. Choose from 'X', 'Y', 'Z'.")
        """

        self.weight_shapes = {"weights": (self.numLayers, self.numWires, 3)}
        

        # Set device for PyTorch
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(self.device)

        
        # Create the quantum node
        self.qnode = self.create_qnode()
        # Define the quantum layer in PyTorch
        self.qlayer = qml.qnn.TorchLayer(self.qnode, self.weight_shapes).to(self.device)
        

        
    
    def create_qnode(self):
        """Creates the quantum node for the hybrid model."""
        @qml.qnode(self.dev)
        def qnode(inputs, weights):
            """# Apply Hadamard if specified
            if self.Hadamard:
                for i in range(self.numWires):
                    qml.Hadamard(wires=i)
            """
            # Encoding and Ansatz logic
            if self.Reuploading:
                for w in weights:
                    self.encodingCircuit(inputs)#qml.AngleEmbedding(inputs, wires=range(self.n_qubits))
                    self.apply_ansatz(w.unsqueeze(0))
            else:
                self.encodingCircuit(inputs)
                self.apply_ansatz(weights)

            # Measurement
            return [qml.expval(qml.PauliZ(wires=i)) for i in range(self.n_outputs)]

        return qnode

    def apply_ansatz(self, weights):
        """Apply the variational circuit."""
        qml.StronglyEntanglingLayers(weights, wires=range(self.numWires))

    def encodingCircuit(self, inputs):
        """
        Apply encoding circuit based on the specified encoding method.
        @ inputs: array of input values in range [-1, 1]
        """
        norms = torch.norm(inputs, dim=1, keepdim=True)
        inputs = inputs / norms
        #inputs_norm = inputs / np.linalg.norm(inputs)
        qml.MottonenStatePreparation(inputs, wires=range(self.numWires))
        """
        for e in self.Encoding:
            qml.AngleEmbedding(math.pi * inputs, wires=range(self.numWires), rotation=e)
        """
    def forward(self, inputs):
        """Forward pass through the hybrid model."""
        return self.qlayer(inputs)
