import pandas as pd
from data_importer import DataImporter

numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

class MyData:
    def __init__(self, filename):
        """
        A class to represent a dataset with basic functionality.

        Parameters:
            filename (str): The name of the file containing the dataset.

        Attributes:
            data_importer (DataImporter): An instance of the DataImporter class used to load the data.
            data (pandas.DataFrame): The loaded dataset.

        """
        self.data_importer = DataImporter(filename)
        self.data = self.data_importer.load_data()

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

    def filter_by_column_value(self, column, value):
        """
        Filters the dataset by the specified column and value.

        Parameters:
            column (str): The name of the column to filter by.
            value: The value to filter by.

        Returns:
            pandas.DataFrame: The filtered dataset.

        """
        return self.data[self.data[column] == value]

    def sort_by_column(self, column, ascending=True):
        """
        Sorts the dataset by the specified column.

        Parameters:
            column (str): The name of the column to sort by.
            ascending (bool): Whether to sort in ascending order. Defaults to True.

        Returns:
            pandas.DataFrame: The sorted dataset.

        """
        return self.data.sort_values(by=column, ascending=ascending)

    def group_by_column(self, column):
        """
        Groups the dataset by the specified column.

        Parameters:
            column (str): The name of the column to group by.

        Returns:
            pandas.core.groupby.DataFrameGroupBy: A grouped dataset.

        """
        return self.data.groupby(column)
    
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
        print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"
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
    
    def replace_column_with_dictionary(self, column, dictionary):
        """
        Replaces the values of the specified column with the specified dictionary.
        Searches the dictionary values in specified column and replaces them with the dictionary keys.

        Parameters:
            column (str): The name of the column to replace values of.
            dictionary (dict): The dictionary to replace the values with. 
            
        """
        for new_value, old_value in dictionary.items():
            self.data[column] = self.data[column].replace_by_value(old_value, new_value)
        
        
    def replace_by_value(self, column, key, val):
        """
        Replaces all occurrences of `key` in the `column` column of the dataframe with `val`.
        
        Parameters:
        column (str): The name of the column in `self.data` to replace values in
        key (object): The value to replace
        val (object): The value to replace `key` with
        
        Returns:
        pandas.DataFrame: A copy of `df` with the specified replacements made
        """
        # Create a boolean mask of rows where `col` equals `key`
        mask = self.data[column] == key
        
        # Replace the values in `col` where `mask` is `True` with `val`
        self.data.loc[mask, column] = val
        
    def drop_columns(self, columns):
        """
        Drops the specified columns from the dataset.

        Parameters:
            columns (list): The list of columns to drop.

        """
        self.data = self.data.drop(columns=columns)

from sklearn.preprocessing import PolynomialFeatures
from enums import AggregationsTypes
import scipy
       
class MyDataFeaturing(MyData):
    """
    A class to featurize the dataset.
    """
    def __init__(self, filename):
        super().__init__(filename)
        self.data_importer = DataImporter(filename)
        self.data = self.data_importer.load_data() 
        
    def set_features_with_aggregation(self, groupedby_feature, features, aggregations):
        """
        Sets the features of the dataset with aggregation.

        Parameters:
            groupedby_feature (str): The feature to group by.
            features (list): The list of features to set.
            aggregations (enum: AggregationType): The list of aggregations to apply to the features.

        """
        bb_aggregations = {}
        
        aggregation_list = []
        for agg in aggregations:
            if agg == AggregationsTypes.MEAN:
                aggregation_list.append('mean')
            elif agg == AggregationsTypes.MEDIAN:
                aggregation_list.append('median')
            elif agg == AggregationsTypes.MAX:
                aggregation_list.append('max')
            elif agg == AggregationsTypes.MIN:
                aggregation_list.append('min')
            elif agg == AggregationsTypes.COUNT:
                aggregation_list.append('count')
            elif agg == AggregationsTypes.SUM:
                aggregation_list.append('sum')
            elif agg == AggregationsTypes.NUNIQUE:
                aggregation_list.append('nunique')
            elif agg == AggregationsTypes.MODE:
                #todo: verify this works
                aggregation_list.append(lambda x: scipy.stats.mode(x)[0])
            else:
                print("Invalid aggregation type.")
        
        for feature in features:
            bb_aggregations[feature] = aggregation_list

            
        self.data = self.data.groupby(groupedby_feature).agg(bb_aggregations)
        col_name = pd.Index([e[0] + "_" + e[1].upper().replace('<', '').replace('>', '') for e in self.data.columns.tolist()])
        self.data.columns = col_name

        self.data = self.data.reset_index()
        
    def set_features_with_polynomial(self, features, degree):
        """
        Sets the features of the dataset with polynomial.

        Parameters:
            features (list): The list of features to set.
            degree (int): The degree of the polynomial.

        """
        polynomial_features = PolynomialFeatures(degree=degree, include_bias=False)
        poly_features = polynomial_features.fit_transform(self.data[features])
        poly_features = pd.DataFrame(poly_features, columns=polynomial_features.get_feature_names(features))
        poly_features.index = self.data.index
        self.data = pd.concat([self.data, poly_features], axis=1)
        
        
    def set_features_with_polynomial_and_aggregation(self, groupedby_feature, features, aggregations, degree):
        """
        Sets the features of the dataset with polynomial and aggregation.

        Parameters:
            groupedby_feature (str): The feature to group by.
            features (list): The list of features to set.
            aggregations (enum: AggregationType): The list of aggregations to apply to the features.
            degree (int): The degree of the polynomial.

        """
        self.set_features_with_aggregation(groupedby_feature, features, aggregations)
        self.set_features_with_polynomial(features, degree)