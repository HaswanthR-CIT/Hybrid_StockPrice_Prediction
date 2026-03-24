# Quantum LSTM Stock Prediction

A Hybrid Quantum-Classical Machine Learning project that integrates Variational Quantum Circuits (VQC) with Long Short-Term Memory (LSTM) networks to predict stock price movements.

## 📌 Project Overview
- **Goal**: Predict if the stock price will go **UP** or **DOWN** the next day.
- **Data**: Live real-time and historical data (OHLCV) for TSLA.
- **Models**:
  1.  **Classical LSTM**: Standard Deep Learning baseline.
  2.  **Hybrid QLSTM**: VQC (PennyLane) + LSTM (TensorFlow).
  3.  **Live Classical + News**: Live prediction pipeline that offsets the Baseline LSTM score with real-time news sentiment gathered via **Custom Web Scraping** (Google News RSS, past 7 days).

## 📂 Structure
- `data/`: Datasets and plots.
- `notebooks/`: Jupyter notebooks for design and experiments.
- `scripts/`: Python scripts for data pipeline, training, and evaluation.
- `models/`: Saved model weights (`.h5` files).

## 🚀 How to Run

### 1. Setup Environment
```bash
# Clone and enter directory
git clone <repo_url>
cd Stock_Price_Prediction

# Create Virtual Environment
python -m venv venv
.\venv\Scripts\activate   # Windows

# Install Dependencies
pip install -r requirements.txt
```

### 2. Run Data Pipeline
Fetch and preprocess the latest stock data:
```bash
python scripts/fetch_data.py
python scripts/preprocess_data.py
```

### 3. Train Baseline Model
Train the classical LSTM to establish a benchmark:
```bash
python scripts/train_baseline.py
# Model saved to models/baseline_lstm.h5
```

### 4. Verify Quantum Components
Check if the quantum circuit is functioning correctly:
```bash
python scripts/test_qml_components.py
```

### 5. Evaluate & Compare
Generate performance metrics and ROC curves:
```bash
python scripts/evaluate_models.py
# Results saved to models/evaluation_results.txt
# Plot saved to data/roc_comparison.png
```

### 6. Make a Live Prediction (Real-Time)
Fetch today's real-time market data for TSLA and predict tomorrow's movement:
```bash
python scripts/fetch_data.py
python scripts/preprocess_data.py
python scripts/predict_tomorrow.py
```

## ⚠️ Known Limitations
- **Hybrid Training**: Training the full hybrid model (`train_hybrid.py`) requires significant simulation time and may have compatibility issues with Python 3.13/Keras 3. The design is fully documented in `notebooks/hybrid_model.ipynb`.

## 👨‍💻 Author
Haswanth R
