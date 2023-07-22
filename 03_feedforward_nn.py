import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
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

# Step 1: Split the data into training and validation sets
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

# Step 2: Create a simple feedforward neural network model
model = tf.keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(784,)),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(10, activation='softmax')
])

# Step 3: Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Step 4: Fit the model on the training data
history = model.fit(X_train, Y_train, epochs=10, batch_size=32, validation_data=(X_val, Y_val))

# Optionally, you can check the training and validation accuracy
train_predictions = model.predict_classes(X_train)
train_accuracy = accuracy_score(Y_train, train_predictions)
print("Training accuracy:", train_accuracy)

val_predictions = model.predict_classes(X_val)
val_accuracy = accuracy_score(Y_val, val_predictions)
print("Validation accuracy:", val_accuracy)

# Step 5: Make predictions on the X_test data
test_predictions = model.predict_classes(X_test)

# Visualize the predictions on X_test
num_samples_to_visualize = 10
sample_indices = np.random.choice(X_test.shape[0], num_samples_to_visualize, replace=False)

plt.figure(figsize=(12, 6))
for i, index in enumerate(sample_indices):
    plt.subplot(2, 5, i + 1)
    plt.imshow(X_test[index].reshape(28, 28), cmap='gray')
    predicted_label = int(test_predictions[index])
    plt.title(f"Predicted Label: {predicted_label}")
    plt.axis('off')

plt.show()
