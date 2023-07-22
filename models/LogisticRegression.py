import numpy as np
import pandas as pd
import tensorflow as tf
import time


class MultinomialLogisticRegression:
    def __init__(self, num_input_variables, num_output_values):
        """
        Summary:
        Initialize first iteration of (random) parameters


        Need the following information for the dimensions of the parameters:
            - Number of input variables
            - Number of output values (labels / classes)
        """

        self.W = tf.Variable(tf.random.normal([num_input_variables, num_output_values]))
        self.b = tf.Variable(tf.zeros([num_output_values]))

    def one_hot_encoding(self, y):
        """
        Summary:
        Y vector with dimensions N x 1 transformed into N x M_y
        or Y.T 1 x N transformed into M_y x N
        For each observation Yi, there are M_y possible values, thus N rows (observations) M_y columns (possible values)
        if Yi equals 2, the 2nd column of the i'th row will equal 1, (100% probability the value is a 2)
            If transformed, this will be the other way around, 2nd row of the i'th column will be equal to 1

        Returns:
        One hot encoded Y
        Note: Do check if Y is transposed, and if the rows represent the number of observations
        """

        one_hot_y = \
            tf.keras.utils.to_categorical(
                y,
                num_classes=len(np.unique(y))
            )
        return one_hot_y

    def SGD_learning_rate(self, epoch, initial_lr, decay_rate, decay_steps):
        """
        Learning rate schedule function for SGD.

        Args:
        epoch (int): Current epoch number.
        initial_lr (float): Initial learning rate.
        decay_rate (float): Decay rate for the learning rate.
        decay_steps (int): Number of epochs after which the learning rate will decay.

        Returns:
        Updated learning rate.
        """

        updated_lr = initial_lr * decay_rate ** (epoch // decay_steps)
        return updated_lr

    def model(self, x, W, b):
        """
        :param x: Input variables
        :param W: Weight matrices
        :param b: Bias matrices

        :return: Softmax/Predictions/Output of logistic regression model for multiple categories
        """
        x = tf.cast(x, tf.float32)
        return tf.nn.softmax(tf.matmul(x, W) + b)

    def prediction(self, softmax=None):
        """
        :param softmax: Probabilities P(Y=label) for each label
        :return: prediction: max over labels {P(Y=label)}
        """
        return np.argmax(softmax, axis=1)

    def cross_entropy_loss(self, y_one_hot, x, W, b):
        """
        :param y: Dependent variable
        :param y_hat: Predicted variable
        :return: Cross entropy loss
        """
        x = tf.cast(x, tf.float32)
        logits = tf.matmul(x, W) + b
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_one_hot, logits=logits))

    def update_parameters(self, learning_rate=0.01):
        """
        :param learning_rate: step size of gradient descent algorithm
        :return: Updated parameterrs

        Update parameters with Gradient Descent:
            Calculate Gradient matrix of the Loss function w.r.t. the Parameters
            Update Parameters by subtracting learning_rate * Gradient matrix of Loss w.r.t Parameters
                From the previous Parameters

        In short: Update parameters s.t. the gradient -> 0 (global/local minimum)
        """
        return tf.optimizers.SGD(learning_rate)

    def calculate_accuracy(self, softmax, y):
        """
        :param softmax: The probabilities an observation might be one of the labels; P(Yi=labelj) for all j
        :param y: The actual label
        :return: The accuracy (how many labels have been predicted correctly overall)

        In short: This function calculates the Accuracy for all the predictions
        """

        y_one_hot = self.one_hot_encoding(y)
        y_hat = np.argmax(softmax, axis=1)
        y_one_hot = np.argmax(y_one_hot, axis=1)
        accuracy = np.mean(y_hat == y_one_hot)
        accuracy = np.round(accuracy * 100, 1)

        return accuracy

    def calculate_accuracy_per_label(self, softmax, y):
        """
        :param softmax: The probabilities an observation might be one of the labels; P(Yi=labelj) for all j
        :param y: The actual label
        :return: The accuracy per label

        In short: This function calculates the Accuracy per label
        """

        y_one_hot = self.one_hot_encoding(y)
        predicted_labels = np.argmax(softmax, axis=1)
        true_labels = np.argmax(y_one_hot, axis=1)
        unique_labels = np.unique(true_labels)
        accuracy_per_label = {}
        for label in unique_labels:
            mask = true_labels == label
            accuracy = np.mean(predicted_labels[mask] == true_labels[mask])
            accuracy_per_label[label] = accuracy
        accuracy_per_label = {key: f"{round(value * 100, 1)}%" for key, value in accuracy_per_label.items()}

        return accuracy_per_label

    def save_model(self, save_path, W, b):
        """
        :param save_path: path to save the model parameters
        :param W: Final W matrix of the model to be saved
        :param b: Final b matrix of the model to be saved
        """
        checkpoint = tf.train.Checkpoint(weights=W, biases=b)
        checkpoint.save(save_path)

    def train_model(
            self, x=None, y=None, epochs=5, learning_rate=0.01, ith_iteration=50
    ):
        """
        :param x: X matrix, input variables (independent variables)
        :param y: Y matrix, output variable (dependent variables)
        :param epochs: Number of times the model is being trained on the training dataset
        :param learning_rate: The step size with which the Parameters are being updated
        :param ith_iteration: Showing the results in each ith iteration, for example ith_iteration=50, shows all results
                              In each 50th iteration of the training

        In short: Method to calculate the model -> Calculate the loss and accuracy -> Show results
                   -> Update the parameters
        until the gradients of the Loss function w.r.t. the Parameters -> 0 <=> Loss function is minimized w.r.t W & b
                                                                            <=> Predictions are optimized w.r.t. y_true

        """

        # Initials random parameters for the first iteration
        W = self.W                          # Dimensions: Number of labels x Number of variables
        b = self.b                          # Dimensions: Number of labels x Number of observations

        # Define DataFrame to save intermediate Parameters
        parameters = pd.DataFrame(
            columns=['Iteration', 'W', 'b', 'Accuracy', 'Accuracy per label']
        )

        # Define list to save the times each epoch takes to train the model
        total_training_time = [0]

        # Train the model for 'epochs' iterations (by default 5)
        for epoch in range(epochs):
            # One hot encode the Y matrix from (N x 1) to (N x Num_Labels)
            y_one_hot = \
                self.one_hot_encoding(
                    y
                )

            # Check how long it takes to train the model
            start_time_1 = time.time()

            # Define an optimizer object (Calculate the gradients for a given loss function w.r.t. parameters)
            optimizer = self.update_parameters(
                learning_rate
            )

            # Start recording operations for automatic differentiation (gradient calculation)
            with tf.GradientTape() as tape:
                # Operation 1: Model function
                softmax = \
                    self.model(
                        x,
                        W,
                        b
                    )

                # Operation 2: Loss function
                loss = \
                    self.cross_entropy_loss(
                        y_one_hot,
                        x,
                        W,
                        b
                    )



            # Save all Parameters for each iteration
            df_index = len(parameters)
            parameters.loc[df_index] = \
                [
                    epoch+1, W.numpy(), b.numpy(),
                    self.calculate_accuracy(softmax, y), self.calculate_accuracy_per_label(softmax, y)
                 ]

            # Check how long it takes to train the model
            end_time_1 = time.time()

            # Continue checking how much time it takes to train the model
            start_time_2 = time.time()

            # Compute gradients of the loss with respect to the trainable variables (W and b)
            gradients = tape.gradient(loss, [W, b])

            # Use the optimizer to update the trainable variables (W and b) based on the calculated gradients
            optimizer.apply_gradients(zip(gradients, [W, b]))

            # Calculate the predictions from the softmax output
            y_hat = \
                self.prediction(
                    softmax
                )

            # Check time again, to see how much time it took to train the model
            end_time_2 = time.time()

            # Calculate how much time it took in total for one epoch to train the model
            training_time_one_epoch = round((end_time_1 - start_time_1) + (end_time_2 - start_time_2), 4)

            # Add training time to total model training time
            total_training_time.append(training_time_one_epoch)

            # ### ==== HOW TO DO THIS PART ==== (EXTRA)
            # # Make a progress bar to visualize how much time the training takes
            # for second in range(int(training_time_one_epoch*100)):
            #     loading_bar = "[" + "=" * (second+1) + ">" + "." * (training_time_one_epoch - second) + "]"
            #     print(f"\r{loading_bar}", end="\r")

            # Show metrics in every ith_iteration
            if (epoch + 1) % ith_iteration == 0:
                print(
                    f"Training Iteration/Epoch: {epoch + 1}\n"
                    f"Total Model Accuracy: {self.calculate_accuracy(softmax, y)}%\n"
                    f"Accuracy per Label:\n{self.calculate_accuracy_per_label(softmax, y)}\n"
                    f"Predictions: {y_hat[:10]}\n"
                    f"True labels: {y[:10].T}\n"
                    f"Average Loss: {loss}\n"
                    f"Training time for this epoch: {training_time_one_epoch} seconds\n"
                    f"Total training time for {epoch+1} epochs: {round(sum(total_training_time), 2)} seconds\n"
                )

        print(
            f"Model iteration that will be used, is chosen by the maximum Accuracy\n"
            f"Highest Accuracy achieved in model iteration: {parameters['Accuracy'].idxmax() + 1}\n"
            f"Highest accuracy achieved for this model: {parameters['Accuracy'].max()}%\n"
            f"Accuracy per label for this model iteration:\n{parameters.iloc[parameters['Accuracy'].idxmax(), 4]}\n"
        )

        iteration, W, b, accuracy, accuracy_per_label = \
            parameters.iloc[parameters['Accuracy'].idxmax(), :]

        return W, b

    # ---- TEST MODEL ----

    def load_model(self, path_to_model=None):
        with open(path_to_model, 'rb') as file:
            data = np.load(file)
            W = data['weights']
            b = data['biases']
        return W, b

    def test_softmax(self, x_test=None, W=None, b=None):
        x_test = tf.cast(x_test, tf.float32)
        softmax = self.model(x_test, W, b)
        return softmax

    def save_forecasts(self, y_hat=None, path_to_forecast_directory=None, file_name=None, dataset=None):
        if dataset == 'MNIST':
            df = pd.DataFrame(
                {
                    'ImageId': [i + 1 for i in range(28000)],
                    'Label': y_hat
                }
            )

            df.to_csv(
                rf"{path_to_forecast_directory}/{file_name}.csv",
                index=False
            )

    def test_model(self, x=None, path_to_model=None, path_to_forecast_directory=None, file_name=None, dataset='MNIST'):

        # Load the model
        W, b = self.load_model(
            path_to_model
        )

        # Make predictions
        softmax = self.test_softmax(
            x,
            W,
            b
        )

        y_hat = self.prediction(
            softmax
        )

        print(
            f"TEST RESULTS:\n"
            f"Predictions: {y_hat[:10]}\n"
        )

        self.save_forecasts(
            y_hat=y_hat,
            path_to_forecast_directory=path_to_forecast_directory,
            file_name=file_name,
            dataset=dataset
        )

        return y_hat
