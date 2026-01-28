import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import pennylane as qml
from pennylane import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, TimeDistributed, Layer
import argparse

# Usage: python scripts/tuning_experiment.py --lstm_units 64 --q_layers 1
parser = argparse.ArgumentParser()
parser.add_argument('--lstm_units', type=int, default=50)
parser.add_argument('--q_layers', type=int, default=2)
args = parser.parse_args()

print(f"Starting Tuning Experiment: LSTM={args.lstm_units}, QLayers={args.q_layers}")

try:
    X_train = np.load('data/X_train.npy')
    y_train = np.load('data/y_train.npy')
    X_test = np.load('data/X_test.npy')
    y_test = np.load('data/y_test.npy')
    
    # Tiny subset for speed
    X_train = X_train[:32]
    y_train = y_train[:32]
    X_test = X_test[:32]
    y_test = y_test[:32]

    n_qubits = 5 
    n_layers = args.q_layers
    dev = qml.device("default.qubit", wires=n_qubits)

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

    weight_shapes = {"weights": (n_layers, n_qubits)}
    qlayer = QuantumLayer(quantum_circuit, weight_shapes, output_dim=n_qubits)
    
    model = Sequential()
    model.add(TimeDistributed(qlayer, input_shape=(60, n_qubits)))
    model.add(LSTM(args.lstm_units, return_sequences=True))
    model.add(LSTM(args.lstm_units))
    model.add(Dense(1, activation='sigmoid'))
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    history = model.fit(X_train, y_train, epochs=1, batch_size=32, verbose=1)
    
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Result: Loss={loss:.4f}, Accuracy={acc:.4f}")
    
    # Save optimized model
    model_name = f'models/tuned_lstm{args.lstm_units}_q{args.q_layers}.h5'
    os.makedirs('models', exist_ok=True)
    model.save(model_name)
    print(f"Saved tuned model to {model_name}")

except Exception as e:
    print(f"Experiment Failed: {e}")
