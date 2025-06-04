import numpy as np
import matplotlib.pyplot as plt
# Simple Linear Regression
# Sample dataset: area (in 1000s of sq ft) and price (in $1000s)
area = np.array([1, 2, 3, 4, 5])
price = np.array([150, 200, 250, 300, 350])
# Visualize the data
plt.scatter(area, price, color='blue')
plt.xlabel("Area (1000s sq ft)")
plt.ylabel("Price ($1000s)")
plt.title("House Prices - Simple Linear Regression")
plt.show()
# Predict function
def predict(X, w, b):
    return X * w + b
# Mean Squared Error Loss
def compute_loss(X, y, w, b):
    m = len(X)
    y_pred = predict(X, w, b)
    return (1/m) * np.sum((y_pred - y)**2)
# Gradient Descent
def gradient_descent(X, y, w, b, learning_rate):
    m = len(X)
    y_pred = predict(X, w, b)
    dw = (2/m) * np.dot((y_pred - y), X)
    db = (2/m) * np.sum(y_pred - y)
    w -= learning_rate * dw
    b -= learning_rate * db
    return w, b
# Training loop
w = 0.0
b = 0.0
learning_rate = 0.01
epochs = 1000
for i in range(epochs):
    w, b = gradient_descent(area, price, w, b, learning_rate)
    if i % 100 == 0:
        loss = compute_loss(area, price, w, b)
        print(f"Epoch {i}: Loss = {loss:.2f}")
print(f"\nSimple Linear Model: price = {w:.2f} * area + {b:.2f}")
# Plot predictions
plt.scatter(area, price, color='blue', label='Actual')
plt.plot(area, predict(area, w, b), color='red', label='Predicted')
plt.xlabel("Area (1000s sq ft)")
plt.ylabel("Price ($1000s)")
plt.title("Simple Linear Regression Prediction")
plt.legend()
plt.show()
# Multiple Linear Regression ===
# Features: area, bedrooms, age of house
X = np.array([
    [1, 1, 10],
    [2, 2, 5],
    [3, 3, 3],
    [4, 3, 2],
    [5, 4, 1]
])
y = np.array([150, 200, 250, 300, 350])
# Prediction for multiple features
def predict_multi(X, w, b):
    return np.dot(X, w) + b
# Loss for multiple features
def compute_loss_multi(X, y, w, b):
    m = len(X)
    y_pred = predict_multi(X, w, b)
    return (1/m) * np.sum((y_pred - y)**2)
# Gradient descent for multiple features
def gradient_descent_multi(X, y, w, b, learning_rate):
    m = len(X)
    y_pred = predict_multi(X, w, b)
    dw = (2/m) * np.dot(X.T, (y_pred - y))
    db = (2/m) * np.sum(y_pred - y)
    w -= learning_rate * dw
    b -= learning_rate * db
    return w, b
# Initialize weights
w = np.zeros(X.shape[1])
b = 0.0
learning_rate = 0.01
epochs = 1000
# Train model
for i in range(epochs):
    w, b = gradient_descent_multi(X, y, w, b, learning_rate)
    if i % 100 == 0:
        loss = compute_loss_multi(X, y, w, b)
        print(f"Epoch {i} (Multi): Loss = {loss:.2f}")
print(f"\nMultiple Linear Model: price = {w[0]:.2f}*area + {w[1]:.2f}*bedrooms + {w[2]:.2f}*age + {b:.2f}")
