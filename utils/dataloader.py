import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split


def dataloader(test_size=0.2, random_state=42):
    data_train = pd.read_csv("data/train.csv")
    data_test = pd.read_csv("data/test.csv")

    x_train = data_train.drop('label', axis=1).values.astype('float32')
    y_train = data_train['label'].copy()
    x_test = data_test.values.astype('float32')

    x_train = x_train/255
    x_test = x_test/255

    y_train = y_train.to_numpy().reshape(-1, 1)

    encoder = OneHotEncoder()
    y_train = encoder.fit_transform(y_train)
    y_train = y_train.toarray()

    x_train, x_validation, y_train, y_validation = \
        train_test_split(x_train, y_train, test_size=test_size, random_state=random_state)

    print(x_train.shape)
    print(y_train.shape)
    print(x_validation.shape)
    print(y_validation.shape)
    # print(x_test.shape)
    # print(y_train[0:100])

    return x_train, y_train, x_validation, y_validation, x_test
