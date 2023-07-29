from utils import dataloader
from utils import visualization
import numpy as np
import tensorflow as tf

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

# Normalize the data
x_train, x_test = \
    dataloader.normalize_data(
        x_train,
        x_test
    )

# Split the data into a training and validation set
x_train, x_validation = \
    dataloader.train_test_split(
        x_train,
    )

y_train, y_validation = \
    dataloader.train_test_split(
        y_train
    )

# Reshape the data for CNN (28x28)
x_train, x_validation = \
    dataloader.reshape_data(
        x_train,
        x_validation
    )

# ---- CNN model using tensorflow ----
# Type of NN
model = tf.keras.Sequential()

# Add Convolutional layer, take pieces of the image and try to find characteristics/features
model.add(
    tf.keras.layers.Conv2D(
        filters=64,
        kernel_size=(5, 5),
        padding='Same',
        activation='relu',
        input_shape=(28, 28, 1)
    )
)


# Keep layer output mean close to 0 and std. close to 1, for faster computations
model.add(
    tf.keras.layers.BatchNormalization()
)

# Add second Convolutional layer, take pieces of the image and try to find characteristics/features
model.add(
    tf.keras.layers.Conv2D(
        filters=64,
        kernel_size=(5, 5),
        padding='Same',
        activation='relu',
        input_shape=(28, 28, 1)
    )
)
# Keep layer output mean close to 0 and std. close to 1, for faster computations
model.add(
    tf.keras.layers.BatchNormalization()
)

# Get most important features
model.add(
    tf.keras.layers.MaxPool2D(
        pool_size=(2, 2)
    )
)

# Add third convolutional layer, connecting features to each other
model.add(
    tf.keras.layers.Conv2D(
        filters=64,
        kernel_size=(3, 3),
        padding='Same',
        activation='relu'
    )
)
model.add(
    tf.keras.layers.BatchNormalization()
)

# Add fourth convolutional layer, to make sure all features get connected to each other eventually
model.add(
    tf.keras.layers.Conv2D(
        filters=64,
        kernel_size=(3, 3),
        padding='Same',
        activation='relu'
    )
)
model.add(
    tf.keras.layers.BatchNormalization()
)

# Get most important features
model.add(
    tf.keras.layers.MaxPool2D(
        pool_size=(2, 2),
        strides=(2, 2)
    )
)

# Add fifth convolutional layer, to make sure all features get connected to each other eventually
model.add(
    tf.keras.layers.Conv2D(
        filters=64,
        kernel_size=(3, 3),
        padding='Same',
        activation='relu'
    )
)
model.add(
    tf.keras.layers.BatchNormalization()
)

# Flatten the features now, this way the labels can be connected to different features and it can be classified
model.add(
    tf.keras.layers.Flatten()
)

# Add normal feed forward layer
model.add(
    tf.keras.layers.Dense(
        256,
        activation='relu'
    )
)
model.add(
    tf.keras.layers.BatchNormalization()
)

# Add final layer, prediction layer
model.add(
    tf.keras.layers.Dense(
        10,
        activation='softmax'
    )
)

# # Summary of model
# model.summary()
# tf.keras.utils.plot_model(
#     model,
#     "model_conv.png",
#     show_shapes=True
# )

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy'],

)

# ---- TRAIN MODEL ----
epochs = 2
y_train = dataloader.to_categorical(
    y_train
)
y_validation = dataloader.to_categorical(
    y_validation
)

history = \
    model.fit(
        x_train,
        y_train,
        epochs=epochs,
        validation_data=(x_validation, y_validation),
        verbose=2  # Defines in how much detail the progress is shown during training
    )

model.save(
    r'/Users/teleradio/Desktop/GitHub/MachineLearning/MNIST_classification/saved_models/CNN_model_2_epochs'
)
