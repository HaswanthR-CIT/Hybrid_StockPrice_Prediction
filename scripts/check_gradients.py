import pennylane as qml
from pennylane import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer
import os

print("Performance & Gradient Check...")

try:
    # 1. Custom Quantum Layer (redefined for standalone script)
    n_qubits = 5 
    n_layers = 2
    dev = qml.device("default.qubit", wires=n_qubits)

    # IMPT: Set interface='tf' for TensorFlow backpropagation
    @qml.qnode(dev, interface="tf")
    def quantum_circuit(inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(n_qubits))
        qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
        return [qml.expval(qml.PauliZ(w)) for w in range(n_qubits)]

    class QuantumLayer(Layer):
        def __init__(self, qnode, weight_shapes, output_dim, **kwargs):
            super().__init__(**kwargs)
            self.qnode = qnode
            self.output_dim = output_dim
            self.w = self.add_weight(shape=weight_shapes["weights"], initializer="random_normal", trainable=True)
        def call(self, inputs):
            return tf.convert_to_tensor(self.qnode(inputs, self.w))

    # 2. Build Mini Model for Gradient Check
    weight_shapes = {"weights": (n_layers, n_qubits)}
    qlayer = QuantumLayer(quantum_circuit, weight_shapes, output_dim=n_qubits)
    
    # Simple model mapping input -> VQC -> output
    inputs = tf.random.uniform((1, n_qubits))
    
    with tf.GradientTape() as tape:
        output = qlayer(inputs)
        loss = tf.reduce_sum(output) # Dummy loss
        
    gradients = tape.gradient(loss, qlayer.trainable_variables)
    
    print("\nGradient Check Results:")
    for var, grad in zip(qlayer.trainable_variables, gradients):
        if grad is None:
             print(f"Variable: {var.name} - Gradient is None!")
        else:
            norm = tf.norm(grad).numpy()
            print(f"Variable: {var.name}, Gradient Norm: {norm}")
            if norm == 0:
                print("WARNING: Zero gradient detected!")
            else:
                print("SUCCESS: Non-zero gradient flowing.")

except Exception as e:
    import traceback
    traceback.print_exc()
