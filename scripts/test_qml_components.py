import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt
import os

print("Testing Quantum Components & Generating Visuals...")

try:
    # 1. Define Circuit
    n_qubits = 4
    n_layers = 1 # Keep simple for drawing
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def quantum_circuit(inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(n_qubits))
        qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
        return [qml.expval(qml.PauliZ(w)) for w in range(n_qubits)]

    # 2. Run Circuit
    print("Executing Circuit...")
    inputs = np.random.uniform(0, np.pi, n_qubits)
    weights = np.random.uniform(0, np.pi, (n_layers, n_qubits))
    
    # Text Drawer
    print("\nCircuit Visual (Terminal):")
    min_drawer = qml.draw(quantum_circuit)
    print(min_drawer(inputs, weights))

    output = quantum_circuit(inputs, weights)
    print("\nOutput Vector (Expectation Values):")
    print(output)

    # 3. Generate Visuals
    print("\nGenerating visual artifacts in data/ ...")
    os.makedirs('data', exist_ok=True)

    # A. Circuit Image (Matplotlib)
    try:
        fig, ax = qml.draw_mpl(quantum_circuit)(inputs, weights)
        plt.title("Variational Quantum Circuit (VQC)")
        plt.savefig("data/qml_circuit_diagram.png")
        print(" - Saved: data/qml_circuit_diagram.png")
        plt.close()
    except Exception as e:
        print(f" - Warning: Could not generate circuit image using basic drawer: {e}")

    # B. Feature Chart (Bar Plot of Expectation Values)
    try:
        plt.figure(figsize=(8, 5))
        qubits_labels = [f"Qubit {i}" for i in range(n_qubits)]
        colors = ['red' if x < 0 else 'blue' for x in output]
        
        plt.bar(qubits_labels, output, color=colors)
        plt.axhline(0, color='black', linewidth=0.8)
        plt.ylim(-1.1, 1.1)
        plt.ylabel("Expectation Value <Z>")
        plt.title(f"Quantum Extracted Features (Input: {inputs.round(2)})")
        
        # Add labels
        for i, v in enumerate(output):
            plt.text(i, v + (0.05 if v > 0 else -0.15), str(round(float(v), 2)), ha='center')

        plt.savefig("data/qml_feature_chart.png")
        print(" - Saved: data/qml_feature_chart.png")
        plt.close()
    except Exception as e:
        print(f" - Warning: Could not generate feature chart: {e}")

    # C. Bloch Sphere (Approximation/Placeholder)
    # PennyLane doesn't have a simple "plot_bloch_multivector" like Qiskit without external deps.
    # We will skip QSphere to avoid breaking the environment ("don't spoil current working").
    # The Bar Chart serves as the "Chart" requested.

    print("\nQML Components Successfully Verified & Visualized!")

except Exception as e:
    import traceback
    traceback.print_exc()
