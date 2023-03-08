import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from data_importer import DataImporter
from custom_logger import get_custom_logger

numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

class MyData:
    def __init__(self, filename, dataname = 'My data', isInfo=False):
        """
        A class to represent a dataset with basic functionality.

        Parameters:
            filename (str): The name of the file containing the dataset.
            dataname (str, optional): The name of the dataset. Defaults to 'My data'.
            isInfo (bool): Whether to print the information on working with MyData class.

        Attributes:
            data_importer (DataImporter): An instance of the DataImporter class used to load the data.
            data (pandas.DataFrame): The loaded dataset.

        """
        self.dataname = dataname
        self.isInfo = isInfo
        self.logger = get_custom_logger()
        self.data_importer = DataImporter(filename)
        self.data = self.data_importer.load_data()
        
        if self.isInfo: self.logger.info('Loaded dataset.')

    def get_summary_statistics(self):
        """
        Returns the summary statistics of the dataset.

        Returns:
            pandas.DataFrame: The summary statistics of the dataset.

        """
        return self.data.describe()

    def get_column_names(self):
        """
        Returns the names of the columns in the dataset.

        Returns:
            list: The names of the columns in the dataset.

        """
        return list(self.data.columns)
    
    def get_number_of_columns(self):
        """
        Returns the number of columns in the dataset.

        Returns:
            int: The number of columns in the dataset.

        """
        return self.data.shape[1]
    
    def get_number_of_rows(self):
        """
        Returns the number of rows in the dataset.

        Returns:
            int: The number of rows in the dataset.

        """
        return self.data.shape[0]
    
    def set_dataset_name(self, name):
        """
        Sets the name of the dataset.

        Parameters:
            name (str): The name of the dataset.

        """
        self.name = name
        
    def get_dataset_name(self):
        """
        Returns the name of the dataset.

        Returns:
            str: The name of the dataset.

        """
        return self.name
    
    def get_numeric_features(self):
        """
        Returns the numeric features of the dataset.

        Returns:
            pandas.DataFrame: The numeric features of the dataset.

        """
        return ', '.join(list((self.data.select_dtypes(include=numerics)).columns))
    
    def get_number_of_numeric_features(self):
        """
        Returns the number of numeric features in the dataset.

        Returns:
            int: The number of numeric features in the dataset.

        """
        return len((self.data.select_dtypes(include=numerics)).columns)
    
    def get_categorical_features(self):
        """
        Returns the categorical features of the dataset.

        Returns:
            pandas.DataFrame: The categorical features of the dataset.

        """
        return ', '.join(list((self.data.select_dtypes(include='object')).columns))
    
    def get_number_of_categorical_features(self):
        """
        Returns the number of categorical features in the dataset.

        Returns:
            int: The number of categorical features in the dataset.

        """
        return len((self.data.select_dtypes(include='object')).columns)
    
    def get_boolean_features(self):
        """
        Returns the boolean features of the dataset.

        Returns:
            pandas.DataFrame: The boolean features of the dataset.

        """
        return ', '.join(list((self.data.select_dtypes(include='bool')).columns))
    
    def get_number_of_boolean_features(self):
        """
        Returns the number of boolean features in the dataset.

        Returns:
            int: The number of boolean features in the dataset.

        """
        return len((self.data.select_dtypes(include='bool')).columns)

    # Function to calculate missing values by column# Funct
    def get_missing_values_table(self):
        # Total missing values
        mis_val = self.data.isnull().sum()

        # Percentage of missing values
        mis_val_percent = 100 * self.data.isnull().sum() / len(self.data)

        # Make a table with the results
        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)

        # Rename the columns
        mis_val_table_ren_columns = mis_val_table.rename(
        columns = {0 : 'Missing Values', 1 : '% of Total Values'})

        # Sort the table by percentage of missing descending
        mis_val_table_ren_columns = mis_val_table_ren_columns[
            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)

        # Print some summary information
        if self.isInfo: self.logger.info("Your selected dataframe has " + str(seld.data.shape[1]) + " columns.\n"
            "There are " + str(mis_val_table_ren_columns.shape[0]) +
              " columns that have missing values.")

        # Return the dataframe with missing information
        return mis_val_table_ren_columns
    
    def get_missing_values_by_column(self, column):
        """
        Returns the missing values of the specified column.

        Parameters:
            column (str): The name of the column to get the missing values of.

        Returns:
            pandas.DataFrame: The missing values of the specified column.

        """
        return self.data[column].isnull().sum()
    
    def get_missing_values_by_column_percent(self, column):
        """
        Returns the percentage of missing values of the specified column.

        Parameters:
            column (str): The name of the column to get the percentage of missing values of.

        Returns:
            pandas.DataFrame: The percentage of missing values of the specified column.

        """
        return 100 * self.data[column].isnull().sum() / len(self.data)