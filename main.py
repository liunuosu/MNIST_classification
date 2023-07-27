import datetime
from utils.dataloader import dataloader
from utils.visualization import visualize
from model.logistic_regression import LogRegModel
from model.convolutional_NN import ConvolutionalNN
import time
from pathlib import Path

now = datetime.datetime.now()
model_filename = now.strftime("%Y-%m-%d_%H-%M-%S")
pathdir = Path("weights", model_filename)
pathdir.mkdir(parents=True, exist_ok=True)

n_epoch = 10
batch_size = 512
random_state = 42
split = 0.2

x_train, y_train, x_validation, y_validation, x_test = dataloader(random_state=random_state, test_size=split)

# visualize(x_train)

#logistic_regression = LogRegModel(num_epochs=n_epoch, batch_size=batch_size, random_state=random_state, pathdir=pathdir)
#start = time.time()
#logistic_regression.fit(x_train, y_train, x_validation, y_validation)
#end = time.time()
#total = end - start
#print(f"Training completed in {total} seconds!")
#prediction = logistic_regression.predict(x_test)
#print(prediction[0:100])


ConvolutionalNN = ConvolutionalNN(num_epochs=n_epoch, batch_size=batch_size, random_state=random_state, pathdir=pathdir)
start = time.time()
x_train = x_train.reshape(-1, 28, 28, 1)
x_validation = x_validation.reshape(-1, 28, 28, 1)
ConvolutionalNN.fit(x_train, y_train, x_validation, y_validation)
end = time.time()
total = end - start
print(f"Training completed in {total} seconds!")
x_test = x_test.reshape(-1, 28, 28, 1)
prediction = ConvolutionalNN.predict(x_test)
print(prediction[0:100])
visualize(x_test)
