"""
Data Loader script.
Load the data and preprocess it such that it is applicable for the machine learning models
"""

import pandas as pd
import numpy as np


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

    variables_mean = x_train.mean(axis=1)
    variables_std = np.std(x_train, axis=1)
    normalized_x_train = (x_train - variables_mean[:, np.newaxis]) / variables_std[:, np.newaxis]
    variables_mean = x_test.mean(axis=1)
    variables_std = np.std(x_test, axis=1)
    normalized_x_test = (x_test / variables_mean[:, np.newaxis]) / variables_std[:, np.newaxis]
    return normalized_x_train, normalized_x_test




