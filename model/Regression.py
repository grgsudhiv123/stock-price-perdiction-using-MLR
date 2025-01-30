import numpy as np
import pandas as pd

# Load and clean the dataset
# data = pd.read_csv('./2020-01-01.csv')
# data['Vol'] = data['Vol'].str.replace(',', '').astype(float)
# X = data[['Open', 'High', 'Low', 'Vol']].values  # Input features as numpy array
# y = data['Close'].values  # Target output as numpy array

# Step 1: Manual feature scaling (Standardization)


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
        print(self.means, self.stds)
        return self.transform(X)


# Initialize scaler and scale the features
# scaler = DataScaler()
# X_scaled = scaler.fit_transform(X)

# Step 2: Multi-Linear Regression Model from Scratch


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

            # Optional: print loss every 100 iterations to monitor training
            if i % 100 == 0:
                mse = np.mean((y - y_predicted) ** 2)
                print(f"Iteration {i}: Mean Squared Error = {mse}")

    def predict(self, X):
        # Predict using the learned weights and bias
        print(f"weights: {self.weights}")
        print(f"bias: {self.bias}")
        return np.dot(X, self.weights) + self.bias


# # Step 3: Train the model
# model = MultiLinearRegression(learning_rate=0.01, num_iterations=1000)
# model.fit(X_scaled, y)
# breakpoint()
# # Step 4: Make predictions (example: predict for the first 5 samples)
# predictions = model.predict(X_scaled[:5])
# print(predictions)
