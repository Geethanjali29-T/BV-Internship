# Iris Classification using KNN
import numpy as np
from collections import Counter
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier  # For comparison
# Load the Iris dataset
iris = load_iris()
X = iris.data  # Features
y = iris.target  # Labels
# Split into training and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Euclidean distance function
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))
# KNN classifier (from scratch)
class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = np.array(X)
        self.y_train = np.array(y)

    def predict(self, X):
        predictions = [self._predict(x) for x in X]
        return np.array(predictions)

    def _predict(self, x):
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]
# Test accuracy for different values of K
k_values = range(1, 11)
accuracies = []
for k in k_values:
    model = KNN(k=k)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    acc = accuracy_score(y_test, predictions)
    accuracies.append(acc)
    print(f"K = {k}, Accuracy = {acc:.2f}")
# Plot accuracy vs. K
plt.figure(figsize=(8, 5))
plt.plot(k_values, accuracies, marker='o', color='blue')
plt.title("Accuracy vs. K in KNN (Iris Dataset)")
plt.xlabel("K")
plt.ylabel("Accuracy")
plt.xticks(k_values)
plt.grid(True)
plt.show()
# Use the best K for final prediction
best_k = k_values[np.argmax(accuracies)]
model = KNN(k=best_k)
model.fit(X_train, y_train)
final_predictions = model.predict(X_test)
print(f"\nBest K: {best_k}")
print(f"Final Accuracy with K={best_k}: {accuracy_score(y_test, final_predictions):.2f}")
# Compare with sklearn's KNN
knn_sklearn = KNeighborsClassifier(n_neighbors=best_k)
knn_sklearn.fit(X_train, y_train)
y_pred_sklearn = knn_sklearn.predict(X_test)
print("Sklearn KNN Accuracy:", accuracy_score(y_test, y_pred_sklearn))
