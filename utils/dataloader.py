"""
Data Loader script.
Load the data and preprocess it such that it is applicable for the machine learning models
"""

import pandas as pd
import numpy as np
from keras.utils import to_categorical


# ---- ADD DATA SPLIT (VALIDATION/TEST) -----

def export_data(
        file_type, path_to_train_data, path_to_test_data
):
    """
    Summary:
    Export data

    Allowed files:
    Excel file
    CSV file
    Any file that is retrievable pandas.read_table()

    Returns:
    df_train, df_test
    """

    if file_type == 'csv':
        df_train = pd.read_csv(path_to_train_data)
        df_test = pd.read_csv(path_to_test_data)
        return df_train, df_test
    elif file_type == 'excel':
        df_train = pd.read_excel(path_to_train_data)
        df_test = pd.read_excel(path_to_test_data)
        return df_train, df_test
    else:
        print(f"No correct file type give, choose between:\n"
              f"csv;excel\n"
              f"Function that will be used now is pd.read_table(), check if the data is exported correctly")
        df_train = pd.read_table(path_to_train_data)
        df_test = pd.read_table(path_to_test_data)
        return df_train, df_test


def make_matrix(
        df_train, df_test
):
    """
    Summary:
    Make data into a matrix (Numpy array)

    Allowed datatypes:
    DataFrame

    Returns:
    training_data, test_data
    """

    training_data = np.array(df_train)
    test_data = np.array(df_test)
    return training_data, test_data


def split_variables(
        training_data, test_data,
        x_row_i=None, x_row_n=None, x_column_i=1, x_column_m=None,
        y_row_i=None, y_row_n=None, y_column_i=0, y_column_m=0,
):
    """
    Summary:
    Split (training/test) data into x and y variables
    & get the dimensions of the x and y data (matrices)
    By Default:
    All rows are seen as observations, and columns as variables
    y variable is all rows and only the first column
    x variable is all rows and 2nd column until the last column

    Allowed data types:
    DataFrame and NumPy array

    Returns:
    x_train, y_train, x_test, y_test, N_training_rows, M_x_training_columns, N_test_rows
    """

    # Split the training data into X and Y variables
    x_train = training_data[x_row_i:x_row_n, x_column_i:x_column_m]
    y_train = training_data[y_row_i:y_row_n, y_column_i:y_column_m]

    # Test data only contains x variables (pixels) and no y variable (labels)
    x_test = test_data

    # Get Matrix dimensions of variables
    N_training_rows, M_x_training_columns = x_train.shape
    N_test_rows = x_test.shape[0]

    return x_train, y_train, x_test, N_training_rows, M_x_training_columns, N_test_rows

def normalize_data(x_train, x_test):
    """
    Summary:
    This normalizes the data, s.t. the model will not have to deal with very large or near 0 numbers:

    Allowed input:
    DataFrame
    NumPy array

    Returns:
    normalized_x_train, normalized_x_test
    normalized DataFrames or NumPy arrays
    """

    normalized_x_train = x_train / 255
    normalized_x_test = x_test / 255
    return normalized_x_train, normalized_x_test

def reshape_data(x_train, x_test, l=28, w=28):
    """

    :param x_train: training data
    :param x_test: test data
    :param l: length of image (number of pixels)
    :param w: width of image (number of pixels)
    :return: reshaped data (can be flattened or made into a 2D image again)
    """
    num_train_samples = x_train.shape[0]
    num_test_samples = x_test.shape[0]
    reshaped_x_train = x_train.reshape((num_train_samples, l, w))
    reshaped_x_test = x_test.reshape((num_test_samples, l, w))
    reshaped_x_train = np.expand_dims(reshaped_x_train, axis=-1)
    reshaped_x_test = np.expand_dims(reshaped_x_test, axis=-1)
    return reshaped_x_train, reshaped_x_test


def train_test_split(x, prop_train_size=0.8):
    observations = x.shape[0]
    sample_size = int(observations * prop_train_size)
    x_train = x[:sample_size, :]
    x_test = x[sample_size:, :]
    return x_train, x_test

    def one_hot_encoding(y):
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
            to_categorical(
                y,
                num_classes=len(np.unique(y))
            )
        return one_hot_y
