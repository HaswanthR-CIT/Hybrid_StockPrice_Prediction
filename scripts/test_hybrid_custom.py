import pennylane as qml
from pennylane import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, TimeDistributed, Layer
import os

print("Testing Hybrid Model Build (Custom Layer)...")

# 1. Define Device and Circuit
n_qubits = 5 
n_layers = 2
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev)
def quantum_circuit(inputs, weights):
    qml.AngleEmbedding(inputs, wires=range(n_qubits))
    qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(w)) for w in range(n_qubits)]

# 2. Custom Keras Layer for QNode
class QuantumLayer(Layer):
    def __init__(self, qnode, weight_shapes, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.qnode = qnode
        self.output_dim = output_dim
        # Initialize weights
        self.w = self.add_weight(
            shape=weight_shapes["weights"],
            initializer="random_normal",
            trainable=True,
            name="weights"
        )

    def call(self, inputs):
        # Convert inputs to tensor if needed, but TF does this.
        # QNode execution in TF needs appropriate casting.
        # PennyLane qnode interface should be "tf" for backprop, but default.qubit is fine if inputs are TF tensors.
        # We need to ensure QNode handles TF tensors correctly.
        # Usually qml.qnode(dev, interface="tf") is preferred.
        # But for this structural test, we just want to see it build.
        
        # Note: calling qnode(inputs, self.w) inside tf.function/graph might require logic.
        # For simple build test:
        return tf.convert_to_tensor(self.qnode(inputs, self.w))

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

# 3. Model Builder
def create_hybrid_model(input_shape):
    weight_shapes = {"weights": (n_layers, n_qubits)}
    
    # Use our Custom Layer
    qlayer = QuantumLayer(quantum_circuit, weight_shapes, output_dim=n_qubits)
    
    model = Sequential()
    # TimeDistributed should work with custom layer if output shape logic is simple
    try:
        model.add(TimeDistributed(qlayer, input_shape=input_shape))
    except Exception as e:
        print(f"TimeDistributed error: {e}")
        # Fallback: Apply to flat input just to prove layer works
        model.add(qlayer)
        
    model.add(LSTM(50, return_sequences=True))
    model.add(LSTM(50))
    model.add(Dense(1, activation='sigmoid'))
    return model

try:
    input_shape = (60, 5)
    model = create_hybrid_model(input_shape)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    model.summary()
    print("\nHybrid Model successfully built and compiled with Custom QuantumLayer.")

except Exception as e:
    import traceback
    traceback.print_exc()
