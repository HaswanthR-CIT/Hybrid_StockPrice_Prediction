# Q&A: Quantum LSTM Stock Prediction


## 1. Project Overview & Steps
**Q: Explain this project from scratch to end. What steps did you take?**

**A:** "The goal was to build a stock price predictor that outperforms classical models by using Quantum Machine Learning (QML).
My workflow had 6 phases:
1.  **Environment Setup**: I set up a specialized Python environment with `PennyLane` (for quantum) and `TensorFlow` (for classical DL).
2.  **Data Prep**: I fetched 5 years of AAPL/TSLA data, normalized it (0-1), and created 60-day sequences (sliding windows) to forecast the next day's movement.
3.  **Baseline Model**: I built a standard classical LSTM network first to set a benchmark score.
4.  **Quantum Design**: I designed a Variational Quantum Circuit (VQC) that takes the 5 stock features (Open, High, Low, Close, Volume) and processes them using qubits.
5.  **Hybrid Integration**: I created a custom Keras layer to inject this Quantum Circuit *before* the LSTM. The data flows: `Input -> Quantum Layer -> LSTM -> Output`.
6.  **Evaluation**: finally, I compared the accuracy and ROC curves of the Classical vs. Hybrid models."

---

## 2. Technical Justifications
**Q: Why did you choose LSTM (Long Short-Term Memory)? Why not CNN or simple RNN?**

**A:** "Stock data is **Time-Series** data; the order of days matters.
- **Simple RNNs** suffer from the 'Vanishing Gradient' problem (they forget long-term trends).
- **CNNs** are good for images but less intuitive for strict temporal sequences.
- **LSTMs** have specialized 'Memory Cells' (gates) that can learn to keep important information for long periods and forget noise. This makes them the industry standard for financial forecasting."

**Q: Why can't this be done in a purely Classical way?**

**A:** "It *can* be done classically (that's my Baseline model). However, classical models might struggle to find non-linear, complex correlations between features.
**The Hypothesis**: Quantum computers process information in a higher-dimensional space (Hilbert Space). By mapping our data into this quantum space, we hope to separate complex patterns that classical linear layers miss, essentially 'enriching' the features before the LSTM sees them."

---

## 3. The "Quantum" Aspect
**Q: How does Quantum come into this? Is this 'purely' a quantum project?**

**A:** "No, this is a **Hybrid** model.
- **Why Hybrid?**: Pure quantum computers (NISQ era) are too small and noisy to run a full Deep Learning network. Classical computers are robust.
- **The Solution**: We use the Quantum device for what it's best at—**Feature Extraction** (finding patterns)—and the Classical LSTM for what it's best at—**Sequence Learning**. The Quantum part acts like a 'smart filter' at the start of the network."

**Q: Explain the Quantum components you used.**

**A:** "I used a Variational Quantum Circuit (VQC) with three main stages:
1.  **Embedding (`AngleEmbedding`)**: This is the 'Input Port'. It takes my 5 classical numbers (Price/Volume) and rotates the 5 qubits to specific angles on the Bloch Sphere. This serves as the data loading step.
2.  **Entanglement (`BasicEntanglerLayers`)**: This is the 'Processing Core'. We apply CNOT gates (Controlled-NOT) to link the qubits together. If 'Open Price' changes, it affects how the network perceives 'Volume'. This **Entanglement** captures deep correlations between features.
3.  **Measurement (`PauliZ`)**: This is the 'Output Port'. We collapse the quantum state back into real numbers (Expectation Values) to feed into the classical LSTM."

**Q: How can you say this is 'Purely Quantum'?**

**A:** "I *wouldn't* say it's 'Purely Quantum'. I would be honest: it is a **Hybrid QLSTM**. It uses valid quantum mechanical principles (Superposition and Entanglement) via the PennyLane simulator to process the data, but the final decision making is shared with classical networks. This is currently the most practical way to use quantum tech today."

---

## 4. Results & Analysis
**Q: Explain your Accuracy and Loss. Is it good?**

**A:**
- **The Metric**: I used Binary Accuracy (Did it go UP or DOWN correctly?).
- **Baseline**: Achieved around 55% accuracy (typical for stock data, which is very noisy/random).
- **Hybrid**: Ideally, we look for an improvement (e.g., 56-60%). Even a 1% improvement in finance is significant.
- **The Difference**: If the Hybrid model's Training Loss drops *faster* or reaches a lower point than the Baseline, it proves the Quantum Features made the learning task easier for the LSTM.

**Q: How could this be improved?**

**A:**
1.  **Real Hardware**: Run the inference on an actual IBM QPU instead of a simulator.
2.  **More Qubits**: Use more than 5 qubits to encode longer time histories (e.g., 10 days of history directly into the quantum state).
3.  **Data Re-uploading**: A more advanced QML technique to load data multiple times for richer expressivity.

---

## 5. Quick Definition Cheat Sheet
- **Qubit**: Quantum bit (0, 1, or both at once).
- **Superposition**: Being in multiple states at once.
- **Entanglement**: Qubits linked together; change one, instantly affect the other.
- **QNode**: A "Quantum Node" - a function in code that runs on a quantum device but looks like a standard Python function.
- **VQC**: Variational Quantum Circuit - A quantum circuit with trainable weights (like neural network weights) that we update using Gradient Descent.
