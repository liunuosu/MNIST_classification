import numpy as np
import tensorflow as tf
from utils import dataloader
from utils import visualization
from models.LogisticRegression import MultinomialLogisticRegression


# ---- EXPORT DATA ----
df_train, df_test = \
    dataloader.export_data(
        'csv',
        r"/Users/teleradio/Desktop/GitHub/MachineLearning/MNIST_classification/data/train.csv",
        r"/Users/teleradio/Desktop/GitHub/MachineLearning/MNIST_classification/data/test.csv"
    )

# ---- DATA PREPROCESSING WITHOUT TENSORFLOW KERAS ----

# Make matrices of data
training_data, test_data = \
    dataloader.make_matrix(
        df_train, df_test
    )

# Split the data in X and Y variables
x_train, y_train, x_test, N_training_rows, M_x_training_columns, N_test_rows = \
    dataloader.split_variables(
        training_data,
        test_data,
        x_row_i=0,
        x_row_n=training_data.shape[0],
        x_column_i=1,
        x_column_m=training_data.shape[1],
        y_row_i=0,
        y_row_n=training_data.shape[0],
        y_column_i=0,
        y_column_m=1
    )

# # Normalize the data
# x_train, x_test = \
#     dataloader.normalize_data(
#         x_train,
#         x_test
#     )

# ---- MODELING WITHOUT TENSORFLOW KERAS ----

# Invoke model instance
model = \
    MultinomialLogisticRegression(
        M_x_training_columns,
        len(np.unique(y_train))
    )

# # Train model
# W, b = \
#     model.train_model(
#         x=x_train,
#         y=y_train,
#         epochs=200,
#         learning_rate=0.01,
#         ith_iteration=1
#     )
#
# # Save model
# with open(
#     r"/Users/teleradio/Desktop/GitHub/MachineLearning/MNIST_classification/models/LogisticRegression_Parameters.npy",
#     'wb'
# ) as file:
#     np.savez(
#         file,
#         weights=W,
#         biases=b
#     )
#
# # ---- TESTING THE MODEL ----
#
# # Test the model
# logistic_regression_model_predictions = \
#     model.test_model(
#         x=x_test,
#         path_to_model=r"/Users/teleradio/Desktop/GitHub/MachineLearning/MNIST_classification/models" +
#                       r"/LogisticRegression_Parameters.npy",
#         path_to_forecast_directory=r"/Users/teleradio/Desktop/GitHub/MachineLearning/MNIST_classification/forecasts/" +
#                                    r"MNIST_forecasts",
#         file_name='submission_logistic_regression'
#     )
#
#
# # ---- VISUALIZE THE MODEL ----
# visualization.show_images(
#     images=x_test,
#     labels=logistic_regression_model_predictions
# )


# ---- DATA PREPROCESSING WITH TENSORFLOW KERAS ----

# Normalize the data
x_train_keras = tf.keras.utils.normalize(x_train, axis=1)
x_test_keras = tf.keras.utils.normalize(x_test, axis=1)


# ---- MODELING WITH TENSORFLOW KERAS ----

# Basic linear combination with non-linearization in layers architecture
tensorflow_keras_logistic_regression_model = \
    tf.keras.models.Sequential()

# Add layers (sigmoid function)
tensorflow_keras_logistic_regression_model.add(
    tf.keras.layers.Dense(10, 'sigmoid')
)

# Softmax function
tensorflow_keras_logistic_regression_model.add(
    tf.keras.layers.Dense(10, 'softmax')
)

# Model optimization configurations
tensorflow_keras_logistic_regression_model.compile(
    optimizer='SGD',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train the model (Fit the model)
tensorflow_keras_logistic_regression_model.fit(
    x_train_keras,
    y_train,
    epochs=200
)

# Save the model
tensorflow_keras_logistic_regression_model.save(
    r'/Users/teleradio/Desktop/GitHub/MachineLearning/MNIST_classification/models/TensorflowKerasModels/LR_model'
)

# Load the model
tensorflow_keras_logistic_regression_model = \
    tf.keras.models.load_model(
        r'/Users/teleradio/Desktop/GitHub/MachineLearning/MNIST_classification/models/TensorflowKerasModels/LR_model'
    )

# Predict the validation data
softmax = \
    tensorflow_keras_logistic_regression_model.predict(
        x_test_keras
    )

y_hat = \
    np.argmax(
        softmax,
        axis=1
    )

# Save the forecasts
model = \
    MultinomialLogisticRegression(
        M_x_training_columns,
        len(np.unique(y_train))
    )

model.save_forecasts(
    y_hat=y_hat,
    path_to_forecast_directory=r"/Users/teleradio/Desktop/GitHub/MachineLearning/MNIST_classification/forecasts/MNIST_forecasts",
    file_name='submission_teras_logistic_regression'
)

loaded_model = \
    tf.keras.models.load_model(
        r"/Users/teleradio/Desktop/GitHub/MachineLearning/MNIST_classification/models/TensorflowKerasModels/LR_model/"
    )

predictions = \
    loaded_model.predict(
        x_test_keras
    )

predicted_classes = \
    np.argmax(
        predictions,
        axis=1
    )

visualization.show_images(
    x_test,
    predicted_classes
)
