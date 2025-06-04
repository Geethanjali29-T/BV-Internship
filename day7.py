import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
# Step 1: Load and Prepare Data
# Load diabetes dataset from sklearn (Note: this is a regression dataset, we simulate classification)
data = load_diabetes()
X = data.data
y_continuous = data.target
# Convert target to binary: 1 if disease risk > 140, else 0
y = (y_continuous > 140).astype(int)
# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# Step 2: Define Logistic Regression from Scratch
# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
# Predict function
def predict_prob(X, w, b):
    return sigmoid(np.dot(X, w) + b)
# Binary cross-entropy loss
def compute_loss(X, y, w, b):
    m = len(y)
    y_pred = predict_prob(X, w, b)
    epsilon = 1e-5  # to avoid log(0)
    loss = - (1/m) * np.sum(y * np.log(y_pred + epsilon) + (1 - y) * np.log(1 - y_pred + epsilon))
    return loss
# Gradient descent
def gradient_descent(X, y, w, b, learning_rate):
    m = len(y)
    y_pred = predict_prob(X, w, b)
    dw = (1/m) * np.dot(X.T, (y_pred - y))
    db = (1/m) * np.sum(y_pred - y)
    w -= learning_rate * dw
    b -= learning_rate * db
    return w, b
# Step 3: Train the Logistic Regression Model
# Initialize parameters
n_features = X_train.shape[1]
w = np.zeros(n_features)
b = 0
learning_rate = 0.1
epochs = 1000
# Training loop
for i in range(epochs):
    w, b = gradient_descent(X_train, y_train, w, b, learning_rate)
    if i % 100 == 0:
        loss = compute_loss(X_train, y_train, w, b)
        print(f"Epoch {i}: Loss = {loss:.4f}")
# Step 4: Make Predictions and Evaluate
# Predict labels
y_probs = predict_prob(X_test, w, b)
y_pred = (y_probs >= 0.5).astype(int)
# Evaluate
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
print(f"\nAccuracy: {acc:.2f}")
print("Confusion Matrix:")
print(cm)
