import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import numpy as np
import os

def create_baseline_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(50))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

if __name__ == "__main__":
    print("Building Baseline LSTM Model...")
    try:
        # Input shape corresponding to our data (60 days, 5 features)
        input_shape = (60, 5)
        model = create_baseline_model(input_shape)
        model.summary()
        
        # Save a dummy summary to a file just to prove it ran if needed, 
        # but stdout is enough for us.
        print("\nModel built and compiled successfully.")
    except Exception as e:
        print(f"Error building model: {e}")
