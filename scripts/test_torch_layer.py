import pennylane as qml
import torch
import traceback

print("Testing TorchLayer Import...")

try:
    from pennylane.qnn import TorchLayer
    print("Success! TorchLayer imported.")
    
    # Create simple torch hybrid to be sure
    n_qubits = 2
    dev = qml.device("default.qubit", wires=n_qubits)
    
    @qml.qnode(dev)
    def qnode(inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(n_qubits))
        qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
        return [qml.expval(qml.PauliZ(w)) for w in range(n_qubits)]
        
    weight_shapes = {"weights": (1, n_qubits)}
    qlayer = TorchLayer(qnode, weight_shapes)
    print("TorchLayer instantiated successfully.")

except Exception:
    with open("error_torch.log", "w") as f:
        f.write(traceback.format_exc())
    print("Error with TorchLayer. Check error_torch.log")
