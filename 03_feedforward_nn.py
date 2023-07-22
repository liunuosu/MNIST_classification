import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers

class NeuralNetworkClassifier:
    def __init__(self, train_file_path, test_file_path):
        self.train_file_path = train_file_path
        self.test_file_path = test_file_path
        self.model = None
        self.history = None

    def load_data(self):
        train = pd.read_csv(self.train_file_path)
        test = pd.read_csv(self.test_file_path)

        X = train.drop(labels=["label"], axis=1).values.astype(np.float32)
        Y = train["label"].values.astype(np.int32)

        X_test = test.values.astype(np.float32)

        X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

        return X_train, X_val, Y_train, Y_val, X_test

    def create_model(self):
        self.model = tf.keras.Sequential([
            layers.Dense(128, activation='relu', input_shape=(784,)),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(10, activation='softmax')
        ])

    def compile_model(self):
        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    def fit_model(self, X_train, Y_train, epochs=10, batch_size=32, validation_data=None):
        self.history = self.model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, validation_data=validation_data)

    def evaluate_model(self, X_train, Y_train, X_val, Y_val):
        train_predictions = self.model.predict_classes(X_train)
        val_predictions = self.model.predict_classes(X_val)

        train_accuracy = accuracy_score(Y_train, train_predictions)
        val_accuracy = accuracy_score(Y_val, val_predictions)

        return train_accuracy, val_accuracy

    def predict(self, X_test):
        return self.model.predict_classes(X_test)

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

    # Create the NeuralNetworkClassifier instance
    nn_classifier = NeuralNetworkClassifier(train_file_path, test_file_path)

    # Load the data
    X_train, X_val, Y_train, Y_val, X_test = nn_classifier.load_data()

    # Create the neural network model
    nn_classifier.create_model()

    # Compile the model
    nn_classifier.compile_model()

    # Fit the model
    nn_classifier.fit_model(X_train, Y_train, epochs=10, batch_size=32, validation_data=(X_val, Y_val))

    # Evaluate the model
    train_accuracy, val_accuracy = nn_classifier.evaluate_model(X_train, Y_train, X_val, Y_val)
    print("Training accuracy:", train_accuracy)
    print("Validation accuracy:", val_accuracy)

    # Make predictions on the test data
    test_predictions = nn_classifier.predict(X_test)

    # Visualize the predictions on the test data
    nn_classifier.visualize_predictions(X_test, test_predictions)
