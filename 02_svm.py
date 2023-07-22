import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import warnings

# Suppress DeprecationWarnings for the entire NumPy module
warnings.filterwarnings("ignore", category=DeprecationWarning, module="numpy")

class SVMClassifier:
    def __init__(self, train_file_path, test_file_path):
        self.train_file_path = train_file_path
        self.test_file_path = test_file_path
        self.svm_model = None

    def load_data(self):
        train = pd.read_csv(self.train_file_path)
        test = pd.read_csv(self.test_file_path)

        X = train.drop(labels=["label"], axis=1).values.astype(np.float32)
        Y = train["label"].values.astype(np.float32)

        X = X[:500]
        Y = Y[:500]

        X_test = test.values.astype(np.float32)

        X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

        return X_train, X_val, Y_train, Y_val, X_test

    def create_model(self):
        self.svm_model = SVC()

    def fit_model(self, X_train, Y_train):
        self.svm_model.fit(X_train, Y_train)

    def evaluate_model(self, X_train, Y_train, X_val, Y_val):
        train_predictions = self.svm_model.predict(X_train)
        val_predictions = self.svm_model.predict(X_val)

        train_accuracy = accuracy_score(Y_train, train_predictions)
        val_accuracy = accuracy_score(Y_val, val_predictions)

        return train_accuracy, val_accuracy

    def predict(self, X_test):
        return self.svm_model.predict(X_test)

    def visualize_predictions(self, X_test, test_predictions, num_samples_to_visualize=10):
        sample_indices = np.random.choice(X_test.shape[0], num_samples_to_visualize, replace=False)

        plt.figure(figsize=(12, 6))
        for i, index in enumerate(sample_indices):
            plt.subplot(2, 5, i + 1)
            plt.imshow(X_test[index].reshape(28, 28), cmap='gray')
            predicted_label = int(test_predictions[index])
            plt.title(f"Predicted Label: {predicted_label}")
            plt.axis('off')

        plt.show()

if __name__ == "__main__":
    # File paths for the dataset
    train_file_path = "data/train.csv"
    test_file_path = "data/test.csv"

    # Create the SVMClassifier instance
    svm_classifier = SVMClassifier(train_file_path, test_file_path)

    # Load the data
    X_train, X_val, Y_train, Y_val, X_test = svm_classifier.load_data()

    # Create the SVM model
    svm_classifier.create_model()

    # Fit the model
    svm_classifier.fit_model(X_train, Y_train)

    # Evaluate the model
    train_accuracy, val_accuracy = svm_classifier.evaluate_model(X_train, Y_train, X_val, Y_val)
    print("Training accuracy:", train_accuracy)
    print("Validation accuracy:", val_accuracy)

    # Make predictions on the test data
    test_predictions = svm_classifier.predict(X_test)

    # Visualize the predictions on the test data
    svm_classifier.visualize_predictions(X_test, test_predictions)
