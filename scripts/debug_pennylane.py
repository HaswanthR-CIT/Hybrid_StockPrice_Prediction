import pennylane as qml
import pennylane.qnn
import sys

print("PennyLane Version:", qml.__version__)
print("QNN module content:", dir(pennylane.qnn))
print("path:", pennylane.qnn.__file__)

try:
    import tensorflow as tf
    print("TensorFlow Version:", tf.__version__)
    print("Keras Version:", tf.keras.__version__)
except Exception as e:
    print("TensorFlow import error:", e)
