from mydata import MyData
from custom_logger import get_custom_logger
import numpy as np
import pandas as pd

class DataProcessing():
    def __init__(self, data:MyData=None, isInfo=True):
        """
        DataProcessing class.

        Parameters:
            data (MyData): The dataset to process. Defaults to None.
            isInfo (bool): Whether to log info messages. Defaults to True.
        """
        self.logger = get_custom_logger()
        self.data = data
        self.isInfo = isInfo
        if self.isInfo: self.logger.info('DataProcessing initialized.')
        
        
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
        
    def reduce_dataframe_memory_usage(self, high_precision = False):
        """
        Iterate through all the columns of a dataframe and modify the data type to
        reduce memory usage.
        Args:
            high_precision (bool): If True, use 64-bit floats instead of 32-bit
        Returns:
            pd.DataFrame: dataframe with reduced memory usage.
        """
        start_mem = round(self.data.memory_usage().sum() / 1024 ** 2, 2)
        if self.isInfo: self.logger.info("Memory usage of dataframe is {0} MB".format(start_mem)) #logging.info
        
        # Iterate through columns
        for col in self.data.columns:
            if self.data[col].dtype == "object":
                # "object" dtype
                if self.data[col].nunique() < max(100, self.data.shape[0] / 100):
                    # If number of unique values is less than max(100, 1%)
                    self.data[col] = self.data[col].astype("category")
                else:
                    # If number of unique values is greater than max(100, 1%)
                    self.data[col] = self.data[col].astype("string")

            elif str(self.data[col].dtype)[:3] == "int":
                # "int" dtype
                c_min = self.data[col].min()
                c_max = self.data[col].max()
                if c_min > np.iinfo(np.uint8).min and c_max < np.iinfo(np.uint8).max:
                    self.data[col] = df[col].astype("UInt8")
                elif c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    self.data[col] = self.data[col].astype("Int8")
                elif c_min > np.iinfo(np.uint16).min and c_max < np.iinfo(np.uint16).max:
                    self.data[col] = self.data[col].astype("UInt16")
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    self.data[col] = self.data[col].astype("Int16")
                elif c_min > np.iinfo(np.uint32).min and c_max < np.iinfo(np.uint32).max:
                    self.data[col] = self.data[col].astype("UInt32")
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    self.data[col] = self.data[col].astype("Int32")
                elif c_min > np.iinfo(np.uint64).min and c_max < np.iinfo(np.uint64).max:
                    self.data[col] = self.data[col].astype("UInt64")
                else:
                    self.data[col] = self.data[col].astype("Int64")

            elif str(self.data[col].dtype)[:5] == "float":
                # "float" dtype
                c_min = self.data[col].min()
                c_max = self.data[col].max()
                if (
                    not high_precision
                    and c_min > np.finfo(np.float32).min
                    and c_max < np.finfo(np.float32).max
                ):
                    self.data[col] = self.data[col].astype("float32")
                else:
                    self.data[col] = self.data[col].astype("float64")

        end_mem = round(self.data.memory_usage().sum() / 1024 ** 2, 2)
        if self.isInfo: self.logger.info("Memory usage after reduction is {0} MB".format(end_mem)) #logging.info
        if (self.isInfo & start_mem > 0) : self.logger.info("Memory usage decreased from {0} MB to {1} MB ({2} % reduction)".format(start_mem, end_mem, round(100 * (start_mem - end_mem) / start_mem, 2))) #logging.info
    
    
    def encode_categorical_variables(self, nan_as_category = True):
        original_columns = list(self.data.columns)
        categorical_columns = [col for col in self.data.columns if self.data[col].dtype == 'object']

        le = LabelEncoder()
        le_count = 0


        for col in categorical_columns:
            if len(list(self.data[col].unique())) <= 2:
                # Train on the training data
                le.fit(self.data[col])
                # Transform both training and testing data
                self.data[col] = le.transform(self.data[col])


        # one-hot encoding of categorical variables
        # Use dummies if > 2 values in the categorical variable
        self.data = pd.get_dummies(self.data, dummy_na = nan_as_category)

        new_columns = [c for c in self.data.columns if c not in original_columns]

        return new_columns
    
    def convert_to_numeric(self, column):
        """
        Converts the specified column of the DataFrame to numeric values.

        Args:
        column (str): The name of the column to be converted.
        """
        self.data[column] = pd.to_numeric(self.data[column], errors="coerce")

    def encode_categorical(self, column):
        """
        Encodes categorical data in the specified column of the DataFrame as numeric values.

        Args:
        column (str): The name of the column to be encoded.
        """
        categories = self.data[column].unique()
        mapping = {category: i for i, category in enumerate(categories)}
        self.data[column] = self.data[column].replace(mapping)
        
        

from sklearn.preprocessing import PolynomialFeatures
from enums import AggregationsTypes
import scipy
       
class DataFeaturing(DataProcessing):
    def __init__(self, data:MyData=None):
        super().__init__(data)
        
        
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