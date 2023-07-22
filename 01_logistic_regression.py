import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import warnings
import time

# Suppress DeprecationWarnings for the entire NumPy module
warnings.filterwarnings("ignore", category=DeprecationWarning, module="numpy")

class LogisticRegressionClassifier:
    def __init__(self, train_file_path, test_file_path, max_iter=100):
        self.train_file_path = train_file_path
        self.test_file_path = test_file_path
        self.max_iter = max_iter
        self.logistic_model = None

    def load_data(self):
        train = pd.read_csv(self.train_file_path)
        test = pd.read_csv(self.test_file_path)

        X = train.drop(labels=["label"], axis=1).values.astype(np.float32)/255
        Y = train["label"].values.astype(np.float32)
        X_test = test.values.astype(np.float32)/255

        X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

        return X_train, X_val, Y_train, Y_val, X_test

    def create_model(self):
        self.logistic_model = LogisticRegression(max_iter=self.max_iter)

    def fit_model(self, X_train, Y_train):
        start_time = time.perf_counter()
        self.logistic_model.fit(X_train, Y_train)
        end_time = time.perf_counter()
        fitting_time = end_time - start_time
        return fitting_time

    def evaluate_model(self, X_train, Y_train, X_val, Y_val):
        train_predictions = self.logistic_model.predict(X_train)
        val_predictions = self.logistic_model.predict(X_val)

        train_accuracy = accuracy_score(Y_train, train_predictions)
        val_accuracy = accuracy_score(Y_val, val_predictions)

        return train_accuracy, val_accuracy

    def predict(self, X_test):
        return self.logistic_model.predict(X_test)

    def visualize_predictions(self, X_test, test_predictions, num_samples_to_visualize=10):
        sample_indices = np.random.choice(X_test.shape[0], num_samples_to_visualize, replace=False)

        plt.figure(figsize=(12, 6))
        for i, index in enumerate(sample_indices):
            plt.subplot(2, 5, i + 1)
            X_test*=255
            plt.imshow(X_test[index].reshape(28, 28), cmap='gray')
            predicted_label = int(test_predictions[index])
            plt.title(f"Predicted Label: {predicted_label}")
            plt.axis('off')

        plt.show()

if __name__ == "__main__":
    # File paths for the dataset
    train_file_path = "data/train.csv"
    test_file_path = "data/test.csv"
    epoch = 200

    # Create the DigitClassifier instance
    logistic_classifier = LogisticRegressionClassifier(train_file_path, test_file_path, max_iter=epoch)

    # Load the data
    X_train, X_val, Y_train, Y_val, X_test = logistic_classifier.load_data()

    # Create the logistic regression model
    logistic_classifier.create_model()

    # Fit the model
    fitting_time = logistic_classifier.fit_model(X_train, Y_train)
    print("Time taken for fitting: {:.2f} seconds".format(fitting_time))

    # Evaluate the model
    train_accuracy, val_accuracy = logistic_classifier.evaluate_model(X_train, Y_train, X_val, Y_val)
    print("Training accuracy:", train_accuracy)
    print("Validation accuracy:", val_accuracy)

    # Make predictions on the test data
    test_predictions = logistic_classifier.predict(X_test)

    # Visualize the predictions on the test data
    logistic_classifier.visualize_predictions(X_test, test_predictions)
