
from keras import Sequential
import tensorflow as tf
# from keras.optimizers import Adam
from keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.utils import Progbar
from keras import backend as K
from keras import layers
import numpy as np
import keras
import os
import keras
import pickle
from pathlib import Path
from utils.dataloader import DataLoader
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn import metrics
from keras import regularizers
from keras import layers, models, regularizers
from sklearn import metrics
from pathlib import Path
from keras import Input, Model
import tensorflow
import time


class ConvNet():

    def __init__(self, input_shape,
                    conv_filters: list,
                    conv_kernels: list,                   
                    conv_strides: list):  
        
        self.input_shape = input_shape
        self.conv_filters = conv_filters
        self.conv_kernels = conv_kernels
        self.conv_strides = conv_strides
        self._num_conv_layers = len(conv_filters)

        self._name = ""
        self.weight_initializer =  tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.05, seed=None )
        self.epoch_count = 1
        self.m_auc = 0
        self.patience = 50

        self.train_loss = []
        self.val_loss = []
        self.val_auc = []
        self.best_epoch = 0

        self.model = ResNet(self.input_shape,self.conv_filters,self.conv_kernels,self.conv_strides,self.weight_initializer,self._name).get_model()

    def save(self, save_folder=".",verbose=True):
        if verbose: print("saved:",save_folder)
        self._create_folder_if_it_doesnt_exist(save_folder)
        self._save_parameters(save_folder)
        self._save_weights(save_folder)
    
    def _save_parameters(self, save_folder):
        parameters = [
            self.input_shape,
            self.conv_filters,
            self.conv_kernels,
            self.conv_strides,
        ]
        save_folder = "trained_models"/Path(save_folder)
        save_path = os.path.join(save_folder, "parameters.pkl")
        with open(save_path, "wb") as f:
            pickle.dump(parameters, f)
    
    def _save_weights(self, save_folder):
        save_folder = "trained_models"/Path(save_folder)
        save_path = os.path.join(save_folder, "weights.h5")
        self.model.save_weights(save_path)

    def _create_folder_if_it_doesnt_exist(self, folder):
        folder = "trained_models"/Path(folder)
        if not os.path.exists(folder):
            os.makedirs(folder)

    def load_weights(self, weights_path):
        self.model.load_weights(weights_path)    

    @classmethod
    def load(cls, save_folder="."):
        temp = save_folder
        save_folder = "saved_models"/Path(save_folder); parameters = None;
        for i in range(2):
            try:  
                parameters_path = os.path.join(save_folder, "parameters.pkl")
                with open(parameters_path, "rb") as f:
                    parameters = pickle.load(f);break
            except FileNotFoundError as e: save_folder = "MIR_trained_models"/Path(temp)
        conv_net = ConvNet(*parameters)  # star for positional arguments!
        weights_path = os.path.join(save_folder, "weights.h5")
        conv_net.load_weights(weights_path)
        return conv_net


    def summary(self, save_image=False):
        self.model.summary()
        tensorflow.keras.utils.plot_model(self.model, "model_conv.png", show_shapes=True)

    def compile(self, learning_rate=0.0001):        
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        bce_loss = SparseCategoricalCrossentropy(from_logits=False)
        # bce_loss = tf.keras.losses.Poisson(reduction="auto", name="poisson")
        self.model.compile(optimizer=optimizer, loss=bce_loss,metrics=[keras.metrics.BinaryAccuracy()])# ,self.sklearnAUC],run_eagerly=True) #self.custom_loss)dw  OR  ['accuracy'] for exact matching 
    
    def train_on_batch(self, batch_size, num_epoch):
        metrics_names = ['train loss','mean loss','train_acc','val_loss','mean_val_loss','val_acc'] 
        self.dataloader = DataLoader(batch_size=batch_size,num_epoch=num_epoch)

        self.loss = []
        meanloss = 0

        self.val_loss_m = []
        meanloss_val = 0
        val_loss2 = 0

        total_train_loss = []
        total_val_loss = []
        try:            
            total_train_loss = list(np.load("visualisation/total_train_loss.npy"))
            total_val_loss = list(np.load("visualisation/total_val_loss.npy"))
            print("loaded loss files")
        except:
            print("no file of previous loss yet")

        
        for epoch_nr in range(0, num_epoch):
            pb_i = Progbar(self.dataloader.len_train_data, stateful_metrics=metrics_names)
            # print(self.dataloader.len_train_data)
            total_correct = 0
            total_samples = 0

            total_correct_train = 0
            total_samples_train = 0
            accuracy = 0
            self.dataloader.shuffle_data()
            # asd
            print("\nepoch {}/{}".format(epoch_nr+1,num_epoch))
            self.counter=0
            for batch_nr in range(self.dataloader.nr_batches):
                # try:
                    x_train, y_train = self.dataloader.load_data(batch_nr=batch_nr)
                    loss = self.model.train_on_batch(x_train, y_train) 
                    loss2 = float(str(loss[0]))# [0:15])
                    self.loss.append(loss[0])  
                    
                    meanloss = np.mean(self.loss) 
                    meanloss = float(str(meanloss))#[0:15])


                    if  batch_nr % 6 == 0 :
                        # try:
                        #     self.counter+=1
                        #     x_val= self.dataloader.x_val
                        #     y_val = self.dataloader.y_val
                        #     y_pred = self.model.predict(x_val,verbose=0)
                        #     y_pred_labels = np.argmax(y_pred,axis=1)
                        #     correct_predictions = np.sum(y_pred_labels == y_train)
                        #     total_correct += correct_predictions
                        #     total_samples += len(y_train)
                            
                        #     accuracy = total_correct / total_samples

                        #     # Since y_pred is already a NumPy array, you can directly calculate the loss
                        #     loss_object = SparseCategoricalCrossentropy(from_logits=True)
                        #     val_loss = loss_object(y_val, y_pred).numpy()

                        #     self.val_loss_m.append(val_loss)
                        #     meanloss_val = np.mean(self.val_loss_m)
                        #     # print("val loss 2",val_loss)
                        #     val_loss2 = float(str(val_loss))#)[0:15])
                        #     self.val_loss_m.append(val_loss)                      
                        #     meanloss_val = np.mean(self.val_loss_m)
                        #     meanloss_val = float(str(meanloss_val))#[0:15])
                        # except:
                            pass


                    if batch_nr == 1:
                        # self.counter+=1
                        x_val= self.dataloader.x_val
                        y_val = self.dataloader.y_val
                        y_pred = self.model.predict(x_val,verbose=0)
                        y_pred_labels = np.argmax(y_pred,axis=1)
                        correct_predictions = np.sum(y_pred_labels == y_val)
                        total_correct += correct_predictions
                        total_samples += len(y_val)
                        
                        accuracy = total_correct / total_samples

                        # Since y_pred is already a NumPy array, you can directly calculate the loss
                        loss_object = SparseCategoricalCrossentropy(from_logits=True)
                        val_loss = loss_object(y_val, y_pred).numpy()

                        self.val_loss_m.append(val_loss)
                        meanloss_val = np.mean(self.val_loss_m)
                        # print("val loss 2",val_loss)
                        val_loss2 = float(str(val_loss))#)[0:15])
                        self.val_loss_m.append(val_loss)                      
                        meanloss_val = np.mean(self.val_loss_m)
                        meanloss_val = float(str(meanloss_val))#[0:15])   
                    #     total_train_loss.append(loss)
                    #     total_val_loss.append(val_loss)   
                    
                    # train_accuracy = 0
                    
                    # Compute predictions from the model for train data end of epoch
                    y_train_pred = self.model.predict(x_train[:1000],verbose=0)
                    y_train_pred_labels = np.argmax(y_train_pred[:1000], axis=1)
                    correct_predictions_train = np.sum(y_train_pred_labels == y_train[:1000])
                    total_correct_train += correct_predictions_train
                    total_samples_train += len(y_train[:1000])
                    train_accuracy = total_correct_train / total_samples_train

                    values=[('train loss',loss2),("mean loss",meanloss),('train_acc',train_accuracy),("val_loss",val_loss2),("mean_val_loss",meanloss_val),('val_acc',accuracy)]  # add comma after last ) to add another metric!        
                    pb_i.add(batch_size, values=values)

                # except:
                    # pass

            self.dataloader.shuffle_data()
            total_train_loss.append(meanloss)
            total_val_loss.append(meanloss_val)  
            # self.dataloader.shuffle_data()
            # self.dataloader.reset_counter() # makes it work after last epoch    
            skip_n_first_epoch = 2 # prevents model from showing absurd scale of start loss
            vis_len =   (len(total_train_loss)) - skip_n_first_epoch     
            if len(total_train_loss) > skip_n_first_epoch:
                # visualize_loss(total_train_loss[-vis_len ::],total_val_loss[-vis_len::],save=True,model_name=self.name) # adding first is buggy
                pass
        
            if epoch_nr%2 == 0:
                self.save(f"{self._name}-{epoch_nr}-{round(meanloss,5)}-{round(meanloss_val,5)}")
                pass
            self.loss = []
            self.val_loss_m = []
            count_val = 0
            count_val2 = 0
            # pb_i.update(num_epoch)
            # time.sleep(0.5)




class ResNet:
    def __init__(self, input_shape, conv_filters, conv_kernels, conv_strides, weight_initializer,name):
        self.input_shape = input_shape
        self.conv_filters = conv_filters
        self.conv_kernels = conv_kernels
        self.conv_strides = conv_strides
        self.weight_initializer = weight_initializer
        self.name = name
        self.model = self._create_model()

    def _create_input(self):
        return Input(shape=self.input_shape, name=self.name)

    def _conv_block(self, input_layer, i):
        x = layers.Conv2D(self.conv_filters[i], self.conv_kernels[i], padding="same",
                        kernel_initializer=self.weight_initializer, kernel_regularizer=regularizers.l1(1e-6))(input_layer)
        x = layers.Activation("relu")(x)
        x = layers.MaxPooling2D(pool_size=(self.conv_strides[i], self.conv_strides[i]))(x)
        x = layers.Dropout(0.2)(x)

        # Projection shortcut using 1x1 Convolution with strides to match the spatial dimensions
        skip_connection = layers.Conv2D(self.conv_filters[i], (1, 1), padding="same", strides=(self.conv_strides[i], self.conv_strides[i]),
                                kernel_initializer=self.weight_initializer, kernel_regularizer=regularizers.l1(1e-6))(input_layer)

        # Add the residual connection by adding the shortcut to the output of the convolutional block
        x = layers.Add()([skip_connection, x])
        return x

    def _dense_layer(self, x):
        x = layers.Flatten()(x)
        x = layers.Dense(1024, kernel_initializer=self.weight_initializer)(x)
        x = layers.Activation("relu")(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(10)(x)
        return x

    def _create_model(self):
        inputs = self._create_input()
        x = inputs
        for i in range(len(self.conv_filters)):
            x = self._conv_block(x, i)
        x = self._dense_layer(x)
        outputs = layers.Activation("sigmoid")(x)
        model = Model(inputs=inputs, outputs=outputs)
        return model

    def get_model(self):
        return self.model



if __name__=="__main__":
    pass