---
title: Neural Network Scratch
mathjax: true
toc: true
categories:
  - scratch
tags:
  - deep_learning
  - computer_vision
  - scratch
---

# Setup

Importar bibliotecas para carregar dados é essencial para preparar conjuntos de treinamento e teste.

```python
import numpy as np
from sklearn.model_selection import train_test_split
from keras.datasets import mnist
from keras.utils import to_categorical
```

# Implementation Neural Network

```python
class DoomNet:
    def __init__(self, input_size, hidden_size, output_size, std=1e-4):
        self.params = {}
        self.params['W1'] = std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def perceptron(self, X, W1, bias):
        return np.dot(X, W1) + bias

    def relu(self, Z):
        return np.maximum(0, Z)

    def softmax(self, Z):
        max = np.max(Z, axis=1, keepdims=True)
        exp_Z = np.exp(Z - max)
        return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)

    def deriv_relu(self, scores):
        return scores > 0

    def forward_prop(self, W1, b1, W2, b2, X):
        Z1 = X.dot(W1) + b1
        A1 = self.relu(Z1)
        Z2 = A1.dot(W2) + b2
        A2 = self.softmax(Z2)

        return Z1, A1, Z2, A2

    def backward_prop(self, Z1, A1, Z2, A2, W1, W2, X, Y):
        m = X.shape[0]
        error = A2 - Y

        dZ2 = error
        dW2 = 1 / m * np.dot(A1.T, dZ2)
        db2 = 1 / m * np.sum(dZ2, axis=0, keepdims=True)
        dZ1 = np.dot(dZ2, W2.T) * self.deriv_relu(Z1)
        dW1 = 1 / m * np.dot(X.T, dZ1)
        db1 = 1 / m * np.sum(dZ1, axis=0, keepdims=True)

        return dW1, db1, dW2, db2

    def cross_entropy_loss(self, A2, Y):
        m = Y.shape[0]
        log_likelihood = -np.log(A2[range(m), Y])
        loss = np.sum(log_likelihood) / m
        return loss

    def update_params(self, W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
        W1 = W1 - alpha * dW1
        b1 = b1 - alpha * db1
        W2 = W2 - alpha * dW2
        b2 = b2 - alpha * db2

        return W1, b1, W2, b2

    def get_predictions(self, A2):
        return np.argmax(A2, axis=1)

    def accuracy(self, Y_pred, Y_true):
        accuracy = np.mean(Y_pred == Y_true)
        return accuracy

    def gradient_desc(self, X, Y, iterations, alpha):
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']

        for i in range(iterations):
            print('Linhas:', X.shape[0])
            Z1, A1, Z2, A2 = self.forward_prop(W1, b1, W2, b2, X)
            dW1, db1, dW2, db2 = self.backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y)
            W1, b1, W2, b2 = self.update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)

            if i <= iterations:
                predictions = self.get_predictions(A2)
                acc = self.accuracy(predictions, np.argmax(Y, axis=1))
                print("Iteration: ", i, " - Accuracy: ", acc)

        return W1, b1, W2, b2
```

# Loading Data

```python
(X_train, y_train), (X_test, y_test) = mnist.load_data()
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

X_train = X_train.reshape((60000, 784))
X_test = X_test.reshape((10000, 784))

X_train = X_train.astype('float32') / 255
X_test = X_test.astype('float32') / 255
```

# Normalizing Data

```python
X_train = X_train.astype('float32') / 255
X_test = X_test.astype('float32') / 255
```

# Training and Predict

```python
model = DoomNet(784, 20, 10)
model.gradient_desc(X_train, y_train, 2000, 0.20)
```
