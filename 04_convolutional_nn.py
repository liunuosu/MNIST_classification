import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers

# Load the kaggle digit dataset
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

# Convert the DataFrame to NumPy arrays with dtype=np.float32
X = train.drop(labels=["label"], axis=1).values.astype(np.float32)
Y = train["label"].values.astype(np.int32)

X_test = test.values.astype(np.float32)

# Reshape the input data to (num_samples, height, width, num_channels)
X = X.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

# Step 1: Split the data into training and validation sets
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

# Step 2: Create a simple Convolutional Neural Network model
model = tf.keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.3),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Step 3: Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Step 4: Fit the model on the training data
history = model.fit(X_train, Y_train, epochs=10, batch_size=32, validation_data=(X_val, Y_val))

# Step 5: Make predictions on the X_test data
test_predictions = model.predict(X_test)
test_labels = np.argmax(test_predictions, axis=1)  # Convert probabilities to class labels

# Visualize the predictions on X_test
num_samples_to_visualize = 10
sample_indices = np.random.choice(X_test.shape[0], num_samples_to_visualize, replace=False)

plt.figure(figsize=(12, 6))
for i, index in enumerate(sample_indices):
    plt.subplot(2, 5, i + 1)
    plt.imshow(X_test[index].reshape(28, 28), cmap='gray')
    predicted_label = test_labels[index]
    plt.title(f"Predicted Label: {predicted_label}")
    plt.axis('off')

plt.show()
