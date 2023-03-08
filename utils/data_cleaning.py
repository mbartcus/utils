import pandas as pd
import numpy as np

class DataCleaning:
    """
    A class for performing basic data cleaning operations.

    Parameters:
    -----------
    data : pandas.DataFrame
        The data to be cleaned.

    Methods:
    --------
    remove_duplicates:
        Remove duplicate rows from the data.

    remove_outliers:
        Remove outliers from a numerical column in the data.

    fill_missing_values:
        Fill missing values in a column of the data.

    drop_missing_values:
        Drop rows with missing values from the data.

    convert_to_numeric:
        Convert a column to a numeric data type.

    encode_categorical:
        Encode a categorical column using integer encoding.
    """
    def __init__(self, data):
        """
        Initialize a DataCleaning instance with the input data.
        """
        self.data = data

    def remove_duplicates(self):
        """
        Remove duplicate rows from the data.
        """
        self.data.drop_duplicates(inplace=True)

    def remove_outliers(self, column, threshold=3, lower_quantile=0.25, upper_quantile=0.75):
        """
        Removes outliers from the specified column of the DataFrame using the interquartile range (IQR) method.

        Args:
        column (str): The name of the column to be cleaned.
        threshold (float): The number of IQRs beyond which a value is considered an outlier.
        lower_quantile (float): The lower quantile used to calculate the IQR.
        upper_quantile (float): The upper quantile used to calculate the IQR.
        """

        q1 = self.data[column].quantile(lower_quantile)
        q3 = self.data[column].quantile(upper_quantile)
        iqr = q3 - q1
        lower_threshold = q1 - threshold * iqr
        upper_threshold = q3 + threshold * iqr
        self.data = self.data[(self.data[column] >= lower_threshold) & (self.data[column] <= upper_threshold)]

    def fill_missing_values(self, column, value=0, method="mean"):
        """
        Fills missing values in the specified column of the DataFrame.

        Args:
        column (str): The name of the column to be cleaned.
        value (int or float): The value to use for filling missing values when method="value".
        method (str): The method to use for filling missing values. Can be "mean", "median", "mode", or "value".
        """

        if method == "mean":
            self.data[column] = self.data[column].fillna(self.data[column].mean())
        elif method == "median":
            self.data[column] = self.data[column].fillna(self.data[column].median())
        elif method == "mode":
            self.data[column] = self.data[column].fillna(self.data[column].mode().iloc[0])
        elif method == "value":
            self.data[column].fillna(value, inplace=True)

    def drop_missing_values(self):
        """
        Drops rows from the DataFrame that contain missing values.
        """
        self.data = self.data.dropna()

    