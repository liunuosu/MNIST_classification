from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from keras import Sequential
from tensorflow.keras import layers
import numpy as np
import keras
import os
import keras
import pickle
from pathlib import Path
from utils.dataloader import DataLoader
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn import metrics
from tensorflow.keras import regularizers

from sklearn import metrics
from pathlib import Path
import tensorflow as tf

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
        self.weight_initializer = None # tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.05, seed=None )
        self.epoch_count = 1
        self.m_auc = 0
        self.patience = 50

        self.train_loss = []
        self.val_loss = []
        self.val_auc = []
        self.best_epoch = 0

        self._create_model()

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
        import tensorflow 
        self.model.summary()
        tensorflow.keras.utils.plot_model(self.model, "model_conv.png", show_shapes=True)

    def compile(self, learning_rate=0.0001):        
        optimizer = Adam(learning_rate=learning_rate)
        bce_loss = BinaryCrossentropy(from_logits=False)
        # bce_loss = tf.keras.losses.Poisson(reduction="auto", name="poisson")
        self.model.compile(optimizer=optimizer, loss=bce_loss,metrics=[keras.metrics.BinaryAccuracy()])# ,self.sklearnAUC],run_eagerly=True) #self.custom_loss)dw  OR  ['accuracy'] for exact matching 
    
    
    def _create_input(self,model):
        print(self.input_shape)
        model.add(layers.Input(shape=(self.input_shape),name="conv_net_input"))
        return model

    def _conv_block(self,model,i):
        #regularizer was 1e-5 for base and no post process -> regularizer was 1e-6 for with postprocessing due to validation loss viewings # MaYBE TRIE 1e-4 FOR NOPOSTPROCESS
        model.add(layers.Conv2D(self.conv_filters[i],self.conv_kernels[i],padding="same",kernel_initializer=self.weight_initializer,kernel_regularizer=regularizers.l1(1e-6)) ) # lower number (e8 < e6) means less regular #WAS 1e-8 -> used to be 1e-4
        # model.add(layers.Conv2D(self.conv_filters[i],self.conv_kernels[i],padding="same",kernel_initializer=self.weight_initializer) )
        # model.add(layers.BatchNormalization())
        model.add(layers.Activation("relu"))
        model.add(layers.MaxPooling2D(pool_size=(self.conv_strides[i],self.conv_strides[i])))
        model.add(layers.Dropout(0.2)) #0.2
        return model
    
    def _dense_layer(self,model):
        model.add(layers.Flatten())
        model.add(layers.Dense(512,kernel_initializer=self.weight_initializer)) #higher value  (0.1 > 0.001) --> the more regularisation
        model.add(layers.Activation("relu"))
        # model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.5)) #0.3
        model.add(layers.Dense(11))
        return model
    
    def _output_layer(self,model):
        model.add(layers.Activation("sigmoid"))
        return model

    def _create_model(self):
        self.model = Sequential()

        # input block
        self.model = self._create_input(self.model)
       
        # conv blocks
        for i in range(self._num_conv_layers):
            self.model = self._conv_block(self.model,i)
        
        # dense layer
        self.model = self._dense_layer(self.model)
            
        # output layer
        self.model = self._output_layer(self.model)


if __name__=="__main__":
    pass