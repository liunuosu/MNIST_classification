import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import numpy as np
import random

class DataLoader():
    def __init__(self, batch_size, num_epoch) -> None:
        self.batch_size = batch_size
        self.num_epoch = num_epoch
        self.x_train, self.y_train, self.x_val, self.y_val, self.x_test = self._get_train_val_test_split()
        self.len_train_data = len(self.x_train)
        self.len_val_data = len(self.x_val)
        self.batch_nr = 0
        self.nr_batches = int(self.len_train_data/self.batch_size)+0 # +1 to include last batch
        print(self.len_train_data)

    def _get_train_val_test_split(self, test_size=0.2, random_state=10):
        data_train = pd.read_csv("data/train.csv")
        data_test = pd.read_csv("data/test.csv")

        x_train = data_train.drop('label', axis=1).values.astype(np.float32)
        y_train = data_train['label'].values.astype(np.float32)
        x_test = data_test.values.astype(np.float32)

        # reshape to be compatible with cnn
        x_train = x_train.reshape(-1, 28, 28, 1)
        x_test = x_test.reshape(-1, 28, 28, 1)

        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=test_size, random_state=random_state)

        x_train = x_train/255
        x_test = x_test/255   

        return x_train, y_train, x_val, y_val, x_test

    # we only shuffle train data to have a consistent validation loss per epoch
    def shuffle_data(self):
        merge = list(zip(self.x_train,self.y_train)) # make sure we shuffle both x and y evenly
        random.shuffle(merge)
        self.x_train, self.y_train = zip(*merge)
        # print(self.x_train[5][100:200]) # sanity check.
    
    # function that returns shuffled batches of train data
    def load_data(self,batch_nr):
        x_train_b = []
        y_train_b = []
        self.batch_nr = batch_nr
        for i in range(batch_nr,batch_nr+self.batch_size):
            try:
                x_train_b.append(self.x_train[(self.batch_size*batch_nr)+i]) # normally you would have a filelist and you load it from disk here. then you append the data to your x_train/y_train
                y_train_b.append(self.y_train[(self.batch_size*batch_nr)+i])
                # print(x_train_b.shape)
                # print((self.batch_size*batch_nr)+i) # sanity check
            except IndexError:
                # last few data points do not fit in batch
                # try to make some logic work for end of array, this is not working yet.
                # x_train_b.append(np.full_like( self.x_train[(self.batch_size*batch_nr)],-1))  
                # y_train_b.append([-1])
                break
        x_train_b = np.array(x_train_b)
        y_train_b = np.array(y_train_b)
        return x_train_b, y_train_b

    def load_val(self,batch_nr):
        x_val_b = []
        y_val_b = []
        self.batch_nr = batch_nr
        for i in range(batch_nr,batch_nr+self.batch_size):
            try:
                x_val_b.append(self.x_val[(self.batch_size*batch_nr)+i]) # normally you would have a filelist and you load it from disk here. then you append the data to your x_train/y_train
                y_val_b.append(self.y_val[(self.batch_size*batch_nr)+i])
                # print((self.batch_size*batch_nr)+i) # sanity check
            except IndexError:
                # last few data points do not fit in batch
                break
        x_val_b = np.array(x_val_b)
        y_val_b = np.array(y_val_b)
        return x_val_b, y_val_b
        
if __name__ == "__main__":
    c = DataLoader(32,100)
    # c.load_data(0)
    pass