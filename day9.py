# Import necessary libraries
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
# Load the digits dataset
digits = datasets.load_digits()
X, y = digits.data, digits.target
# Visualize some digits
plt.figure(figsize=(10, 4))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(digits.images[i], cmap='gray')
    plt.title(f"Label: {digits.target[i]}")
    plt.axis('off')
plt.tight_layout()
plt.show()
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# Feature scaling (important for SVMs)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# Linear SVM
print("Training Linear SVM...")
linear_svm = SVC(kernel='linear', C=1.0) # C is a hyperparameter
linear_svm.fit(X_train, y_train)
y_pred_linear = linear_svm.predict(X_test)
print("\n=== Linear SVM Classification Report ===")
print(classification_report(y_test, y_pred_linear))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_linear))
# Kernel SVM (RBF Kernel)
print("\nTraining Kernel SVM (RBF)...")
rbf_svm = SVC(kernel='rbf', C=10.0, gamma=0.01) # C and gamma are hyperparameters
rbf_svm.fit(X_train, y_train)
y_pred_rbf = rbf_svm.predict(X_test)
print("\n=== RBF Kernel SVM Classification Report ===")
print(classification_report(y_test, y_pred_rbf))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rbf))
poly_svm = SVC(kernel='poly', degree=3, C=1.0)
poly_svm.fit(X_train, y_train)
print("\n=== Polynomial Kernel SVM Classification Report ===")
print(classification_report(y_test, poly_svm.predict(X_test)))
