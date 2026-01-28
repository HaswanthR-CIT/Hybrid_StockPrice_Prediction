import numpy as np
import matplotlib.pyplot as plt
import os

print("Loading processed data...")
try:
    X_train = np.load('data/X_train.npy')
    y_train = np.load('data/y_train.npy')
    X_test = np.load('data/X_test.npy')
    y_test = np.load('data/y_test.npy')
    
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    
    # Check Class Balance
    unique, counts = np.unique(y_train, return_counts=True)
    balance = dict(zip(unique, counts))
    print(f"Class Balance (Train): {balance}")
    
    # Plot Class Balance
    plt.figure(figsize=(6, 4))
    plt.bar(balance.keys(), balance.values(), color=['red', 'green'])
    plt.xticks([0, 1], ['Down', 'Up'])
    plt.title('Class Balance in Training Set')
    plt.xlabel('Movement')
    plt.ylabel('Count')
    plt.savefig('data/class_balance.png')
    print("Saved class balance plot to data/class_balance.png")
    
    # Basic Stats of Features (Flattened)
    print("\nFeature Statistics (Scaled):")
    print(f"Mean: {np.mean(X_train):.4f}")
    print(f"Std: {np.std(X_train):.4f}")
    print(f"Min: {np.min(X_train):.4f}")
    print(f"Max: {np.max(X_train):.4f}")
    
    # Plot Feature Distributions
    plt.figure(figsize=(10, 6))
    plt.hist(X_train.flatten(), bins=50, color='blue', alpha=0.7)
    plt.title('Feature Distribution (Scaled)')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.savefig('data/feature_dist.png')
    print("Saved feature distribution plot to data/feature_dist.png")

except Exception as e:
    print(f"Error during validation: {e}")
