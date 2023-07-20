import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import warnings

# Suppress DeprecationWarnings for the entire NumPy module
warnings.filterwarnings("ignore", category=DeprecationWarning, module="numpy")

# Load the kaggle digit dataset
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

# Convert the DataFrame to NumPy arrays with dtype=np.float32
X = train.drop(labels=["label"], axis=1).values.astype(np.float32)
Y = train["label"].values.astype(np.float32)

X_test = test.values.astype(np.float32)

# Step 1: Split the data into training and validation sets
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

# Step 2: Create a logistic regression model
logistic_model = LogisticRegression(max_iter=200)

# Step 3: Fit the model on the training data
logistic_model.fit(X_train, Y_train)

# Optionally, you can check the training accuracy
train_predictions = logistic_model.predict(X_train)
train_accuracy = accuracy_score(Y_train, train_predictions)
print("Training accuracy:", train_accuracy)

# Step 4: Make predictions on the validation data
val_predictions = logistic_model.predict(X_val)

# Step 5: Calculate the validation accuracy
val_accuracy = accuracy_score(Y_val, val_predictions)
print("Validation accuracy:", val_accuracy)

# Step 6: Make predictions on the X_test data
test_predictions = logistic_model.predict(X_test)

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
