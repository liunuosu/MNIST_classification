import tensorflow as tf
from utils.metrics import accuracy
from tensorflow.python.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D


class ConvolutionalModel(tf.Module):
    def __init__(self, batch_size, num_classes, seed=42):
        super(ConvolutionalModel, self).__init__()
        self.conv1 = Conv2D(32, 5, activation=tf.nn.relu, input_shape=(batch_size, 28, 28, 1))
        self.conv2 = Conv2D(64, 5, activation=tf.nn.relu)
        self.pool = MaxPooling2D()
        self.flatten = Flatten()
        self.d1 = Dense(1024, activation=tf.nn.relu)
        self.d2 = Dense(num_classes, activation=tf.nn.softmax)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.d1(x)
        return self.d2(x)


def get_class(y_prob):
    return tf.argmax(y_prob, axis=1)


class ConvolutionalNN:
    def __init__(self, num_epochs=50, batch_size=1024, random_state=42, pathdir=None):
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.random_state = random_state
        self.pathdir = pathdir

        self.model = None

    def fit(self, x_train, y_train, x_validation, y_validation, random_state=42):

        self.model = ConvolutionalModel(batch_size=self.batch_size, num_classes=10, seed=random_state)

        checkpoint = tf.train.Checkpoint(model=self.model)
        manager = tf.train.CheckpointManager(checkpoint, directory=self.pathdir, max_to_keep=20)

        num_train_samples, height, width, depth = x_train.shape
        classifier_opt = tf.optimizers.Adam()  # Add learning rate?

        for epoch in range(self.num_epochs):
            shuffled_ids = [i for i in range(num_train_samples)]

            for i in range(num_train_samples // self.batch_size):
                batch_ids = shuffled_ids[self.batch_size * i: self.batch_size * (i + 1)]
                batch_features = x_train[batch_ids].astype('float32')
                batch_labels = y_train[batch_ids].astype('float32')

                with tf.GradientTape() as tape:
                    classifier_logits = self.model.forward(batch_features)

                    loss_classifier = tf.reduce_mean(
                        tf.nn.softmax_cross_entropy_with_logits(labels=batch_labels.astype('float32'),
                                                                logits=classifier_logits))
                gradients = tape.gradient(loss_classifier, self.model.trainable_variables)
                classifier_opt.apply_gradients(zip(gradients, self.model.trainable_variables))

                if i == (num_train_samples // self.batch_size - 1): # == 0 and i != 0:
                    y_prob = self.model.forward(
                        x_train.astype('float32'))

                    train_accuracy = accuracy(get_class(y_prob),
                                              tf.argmax(y_train, axis=1))

                    y_prob_validation= self.model.forward(
                        x_validation.astype('float32'))

                    validation_accuracy = accuracy(get_class(y_prob_validation),
                                                   tf.argmax(y_validation, axis=1))

                    print("(Training Classifier) epoch %d; training accuracy: %f; "
                          "validation accuracy: %f; batch classifier loss: %f" % (
                        epoch+1, train_accuracy, validation_accuracy, loss_classifier))
                    manager.save()
        return self

    def predict(self, x_test):
        y_prob_test = self.model.forward(
            x_test.astype('float32'))
        return get_class(y_prob_test)



