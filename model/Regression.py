import numpy as np
import pandas as pd

class DataScaler:
    def __init__(self, means=None, stds=None):
        self.means = means
        self.stds = stds

    def fit(self, X):
        # Calculate mean and standard deviation for each feature
        self.means = np.mean(X, axis=0)
        self.stds = np.std(X, axis=0)

    def transform(self, X):
        # Standardize features
        return (X - self.means) / self.stds

    def fit_transform(self, X):
        self.fit(X)
        print("Means:", self.means, "\nStds:", self.stds)
        return self.transform(X)

class MultiLinearRegression:
    def __init__(self, learning_rate=0.00000001, num_iterations=1000, weights=None, bias=None):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = weights
        self.bias = bias

    def fit(self, X, y):
        # Initialize weights and bias
        num_samples, num_features = X.shape
        self.weights = np.zeros(num_features)
        self.bias = 0

        # Gradient descent
        for i in range(self.num_iterations):
            # Linear prediction
            y_predicted = np.dot(X, self.weights) + self.bias

            # Calculate gradients
            dw = (1 / num_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / num_samples) * np.sum(y_predicted - y)

            # Update weights and bias
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            # Monitor training every 100 iterations
            if i % 100 == 0:
                mse = np.mean((y - y_predicted) ** 2)
                print(f"Iteration {i}: Mean Squared Error = {mse}")

    def predict(self, X):
        # Predict using the learned weights and bias
        print("weights:", self.weights)
        print("bias:", self.bias)
        return np.dot(X, self.weights) + self.bias