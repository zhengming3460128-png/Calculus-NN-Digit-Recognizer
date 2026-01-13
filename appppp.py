import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml

print("Loading MNIST digits...")
mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
X = (mnist.data / 255.0).astype(np.float32)
y = mnist.target.astype(int)

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))


def sigmoid_derivative(a):
    return a * (1 - a)

input_size = 784
hidden_size = 128
output_size = 10

np.random.seed(42)
W1 = np.random.randn(input_size, hidden_size) * np.sqrt(1. / input_size)
W2 = np.random.randn(hidden_size, output_size) * np.sqrt(1. / hidden_size)

train_size = 50000
epochs = 1000
learning_rate = 0.8
loss_history = []

print(f"Training on {train_size} samples...")

for epoch in range(epochs):
    idx = np.random.choice(train_size, 128)
    X_batch = X[idx]
    Y_batch = np.eye(10)[y[idx]]

    z1 = np.dot(X_batch, W1)
    a1 = sigmoid(z1)
    z2 = np.dot(a1, W2)
    a2 = sigmoid(z2)

    error = Y_batch - a2
    loss = np.mean(np.square(error))
    loss_history.append(loss)

    d_output = error * sigmoid_derivative(a2)
    d_hidden = np.dot(d_output, W2.T) * sigmoid_derivative(a1)

    W2 += np.dot(a1.T, d_output) * (learning_rate / len(idx))
    W1 += np.dot(X_batch.T, d_hidden) * (learning_rate / len(idx))

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.6f}")

test_X = X[20000:22000]
test_y = y[20000:22000]
l1 = sigmoid(np.dot(test_X, W1))
out = sigmoid(np.dot(l1, W2))
predictions = np.argmax(out, axis=1)
accuracy = np.mean(predictions == test_y)
print(f"\n>>> Final Accuracy: {accuracy * 100:.2f}%")
