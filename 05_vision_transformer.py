import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers
from vit_keras import vit, utils
import cv2

# Load the kaggle digit dataset
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

# Convert the DataFrame to NumPy arrays with dtype=np.float32
X = train.drop(labels=["label"], axis=1).values.astype(np.float32)
Y = train["label"].values.astype(np.int32)

X = X[:100]
Y= Y[:100]
# Reshape the input data to (num_samples, height, width, num_channels)
X = X.reshape(-1, 28, 28, 1)

# Convert grayscale images to color images by duplicating the single channel
X = np.repeat(X, 3, axis=-1)

# Resize the images to match the expected input shape of the ViT model
X = np.array([cv2.resize(image, (96, 96)) for image in X])

# Step 1: Split the data into training and validation sets
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

# Step 2: Create a Vision Transformer model
model = vit.vit_b16(
    image_size=96,  # Change the image size to 96x96
    activation='relu',
    pretrained=True,
    include_top=True,
    pretrained_top=False,
    classes=10,
    weights='imagenet21k'
)

# Step 3: Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Step 4: Fit the model on the training data
history = model.fit(X_train, Y_train, epochs=10, batch_size=32, validation_data=(X_val, Y_val))

# Optionally, you can check the training and validation accuracy
train_predictions = model.predict(X_train).argmax(axis=1)
train_accuracy = accuracy_score(Y_train, train_predictions)
print("Training accuracy:", train_accuracy)

val_predictions = model.predict(X_val).argmax(axis=1)
val_accuracy = accuracy_score(Y_val, val_predictions)
print("Validation accuracy:", val_accuracy)

# Step 5: Make predictions on the test data
X_test = test.values.astype(np.float32)
X_test = X_test.reshape(-1, 28, 28, 1)
X_test = np.repeat(X_test, 3, axis=-1)
X_test = np.array([cv2.resize(image, (96, 96)) for image in X_test])
test_predictions = model.predict(X_test).argmax(axis=1)

# Visualize the predictions on X_test
num_samples_to_visualize = 10
sample_indices = np.random.choice(X_test.shape[0], num_samples_to_visualize, replace=False)

plt.figure(figsize=(12, 6))
for i, index in enumerate(sample_indices):
    plt.subplot(2, 5, i + 1)
    plt.imshow(X_test[index], cmap='gray')
    predicted_label = int(test_predictions[index])
    plt.title(f"Predicted Label: {predicted_label}")
    plt.axis('off')

plt.show()
