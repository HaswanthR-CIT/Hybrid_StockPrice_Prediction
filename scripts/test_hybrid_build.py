import os
# Must set this before importing tensorflow/pennylane
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import pennylane as qml
from pennylane import numpy as np
import tensorflow as tf
import traceback

print("Testing Hybrid Model Build (Legacy Keras)...")

try:
    # Try explicit import if the package level import fails
    try:
        from pennylane.qnn import KerasLayer
    except ImportError:
        print("Standard import failed. Trying pennylane.qnn.keras...")
        from pennylane.qnn.keras import KerasLayer

    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, TimeDistributed

    # 1. Define Device and Circuit
    n_qubits = 5 
    n_layers = 2
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def quantum_circuit(inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(n_qubits))
        qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
        return [qml.expval(qml.PauliZ(w)) for w in range(n_qubits)]

    # 2. Define Model Builder
    def create_hybrid_model(input_shape):
        weight_shapes = {"weights": (n_layers, n_qubits)}
        qlayer = KerasLayer(quantum_circuit, weight_shapes, output_dim=n_qubits)
        
        model = Sequential()
        model.add(TimeDistributed(qlayer, input_shape=input_shape))
        model.add(LSTM(50, return_sequences=True))
        model.add(LSTM(50))
        model.add(Dense(1, activation='sigmoid'))
        return model

    # 3. Build and Summarize
    input_shape = (60, 5)
    model = create_hybrid_model(input_shape)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    model.summary()
    print("\nHybrid Model successfully built and compiled.")

except Exception:
    with open("error.log", "w") as f:
        f.write(traceback.format_exc())
    print("Error occurred. Check error.log")
