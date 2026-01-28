import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
import numpy as np
import os
import datetime

print("Starting Baseline Model Training...")

try:
    # 1. Load Data
    X_train = np.load('data/X_train.npy')
    y_train = np.load('data/y_train.npy')
    X_test = np.load('data/X_test.npy')
    y_test = np.load('data/y_test.npy')
    print(f"Data loaded. Train shape: {X_train.shape}")

    # 2. Build Model
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(LSTM(50))
    model.add(Dense(1, activation='sigmoid'))
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    # 3. Callbacks
    log_dir = "logs/fit/baseline/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
    
    early_stopping = EarlyStopping(
        monitor='val_loss', 
        patience=5, 
        restore_best_weights=True,
        verbose=1
    )
    
    os.makedirs('models', exist_ok=True)
    checkpoint = ModelCheckpoint(
        'models/baseline_lstm_best.h5', 
        monitor='val_loss', 
        save_best_only=True,
        verbose=1
    )

    # 4. Train
    # Increase epochs since we have EarlyStopping
    epochs = 50
    history = model.fit(
        X_train, y_train, 
        epochs=epochs, 
        batch_size=32, 
        validation_data=(X_test, y_test),
        callbacks=[early_stopping, tensorboard_callback, checkpoint]
    )
    
    # 5. Final Save
    model.save('models/baseline_lstm.h5')
    print("Final model saved to models/baseline_lstm.h5")

except Exception as e:
    import traceback
    traceback.print_exc()
