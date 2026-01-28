## 1. Project Overview (Show Slides or Readme)
**Hook**: "Traditional financial models fail to capture complex non-linear patterns in stock data. This project proposes a **Quantum-Classical Hybrid** approach, leveraging Quantum Machine Learning (QML) to extract features via entanglement before feeding them into an LSTM."

**Key Points**:
- **Baseline**: Standard LSTM (Classical).
- **Innovation**: VQC (Variational Quantum Circuit) + LSTM (Hybrid).
- **Novelty**: Using `AngleEmbedding` and `BasicEntanglerLayers` for feature enrichment.

## 2. Walkthrough Steps (Live Demo)

### Step 1: Show Project Structure (VSCode)
- Open VSCode. Extoll the clean structure:
  - `data/`: Raw and processed stock data.
  - `notebooks/`: Design lab for experiments.
  - `scripts/`: Production-ready modular code.
  - `models/`: Saved trained weights.

### Step 2: Data Pipeline (Live Run)
**Say**: "First, we fetch and preprocess real market data for Apple and Tesla."
**Run**:
```bash
python scripts/fetch_data.py
python scripts/preprocess_data.py
```
**Show**: Open `data/` folder and point to the newly created `.npy` files.

### Step 3: Classical Baseline Training (Live Run)
**Say**: "We train a classical LSTM baseline to establish a benchmark."
**Run**:
```bash
python scripts/train_baseline.py
```
**Observe**: Show the progress bar training for 50 epochs (or until early stopping). Point out the loss decreasing.

### Step 4: Quantum Components (Live Run)
**Say**: "Here is the Quantum Feature Extractor. We verify the circuit works in isolation and visualize the quantum state."
**Run**:
```bash
python scripts/test_qml_components.py
```
**Observe**: 
1.  **Terminal**: Show the ASCII circuit drawing and the output vector numbers.
2.  **Visuals**: Navigate to the `data/` folder and open:
    - `qml_circuit_diagram.png`: "This is the generated VQC architecture."
    - `qml_feature_chart.png`: "This chart shows the distinct feature values extracted by the quantum measurement."

### Step 5: Hybrid Model Architecture (Show Code)
**Say**: "For the hybrid model, we integrate this circuit layer into Keras."
**Action**: Open `notebooks/hybrid_model.ipynb`.
- Scroll to "Define Quantum Circuit". Explain the 5 qubits (one for each feature: O, H, L, C, V).
- Scroll to "Custom QuantumLayer". Explain that you wrote a custom layer to handle the quantum-classical data handover.
- **Note**: Mention that training this hybrid model requires significant compute time and specific library versions (Python 3.10), so for this demo, we will analyze the pre-computed results.

### Step 6: Evaluation & Results (Show Artifacts)
**Run**:
```bash
python scripts/evaluate_models.py
```
**Action**:
- Open `data/roc_comparison.png` (generated image).
- **Explain**: "The ROC curve shows the model's ability to distinguish Up vs Down movements."
- Open `models/evaluation_results.txt`. Discuss the Accuracy/F1 scores.

