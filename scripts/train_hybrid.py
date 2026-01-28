import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import pennylane as qml
from pennylane import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, TimeDistributed, Layer
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
import datetime

print("Starting Hybrid Model Training (Legacy Keras)...")

try:
    # 1. Load Data
    X_train = np.load('data/X_train.npy')
    y_train = np.load('data/y_train.npy')
    X_test = np.load('data/X_test.npy')
    y_test = np.load('data/y_test.npy')
    
    SUBSET_SIZE = 64
    X_train_small = X_train[:SUBSET_SIZE]
    y_train_small = y_train[:SUBSET_SIZE]
    X_test_small = X_test[:SUBSET_SIZE]
    y_test_small = y_test[:SUBSET_SIZE]

    # 2. Define QML Layer (Custom workaround)
    n_qubits = 5 
    n_layers = 2
    dev = qml.device("default.qubit", wires=n_qubits)

    # Use TF interface
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
        def compute_output_shape(self, input_shape):
            return (input_shape[0], self.output_dim)

    # 3. Build Model
    weight_shapes = {"weights": (n_layers, n_qubits)}
    qlayer = QuantumLayer(quantum_circuit, weight_shapes, output_dim=n_qubits)
    
    model = Sequential()
    model.add(TimeDistributed(qlayer, input_shape=(60, n_qubits)))
    model.add(LSTM(50, return_sequences=True))
    model.add(LSTM(50))
    model.add(Dense(1, activation='sigmoid'))
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    # 4. Train
    print("Training on small subset...")
    model.fit(
        X_train_small, y_train_small, 
        epochs=3, 
        batch_size=32, 
        validation_data=(X_test_small, y_test_small)
    )
    
    # 5. Save
    os.makedirs('models', exist_ok=True)
    model.save('models/qlstm_hybrid.h5')
    print("Hybrid Model saved to models/qlstm_hybrid.h5")

except Exception as e:
    import traceback
    traceback.print_exc()
