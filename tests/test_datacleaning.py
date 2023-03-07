import sys
# import your modules and packages from anywhere, i.e., from any directory on your computer.
sys.path.append('../utils/utils')

import data_cleaning as dc
import pandas as pd
import unittest


class TestDataCleaning(unittest.TestCase):
    
    def setUp(self):
        # Create sample data for testing
        self.initial_data = pd.DataFrame({
            'A': [1, 2, 3000, 3, 4, 5, 6, 7, 8, 9, 9, 10, 2],
            'B': [1, 2, 3, 4, 5, 6, 7, None, 9, 10, 11, 12, 2],
            'C': ['apple', 'orange', 'banana', 'orange', 'apple', 'orange', 'banana', 'orange', 'apple', 'orange', 'banana', 'orange', 'orange']
        })
        
        self.len_data = len(self.initial_data)

        # Instantiate the DataCleaning object with the sample data
        self.dc = dc.DataCleaning(self.initial_data)
    
    def test_remove_duplicates(self):
        # Test removing duplicates
        self.dc.remove_duplicates()
        self.assertEqual(len(self.dc.data),  self.len_data-1)
    
    def test_remove_outliers(self):
        # Test removing outliers
        self.dc.remove_outliers('A')
        self.assertEqual(len(self.dc.data), self.len_data-1)
    
    def test_fill_missing_values(self):
        # Test filling missing values with mean
        self.dc.fill_missing_values('B', method='mean')
        self.assertFalse(self.dc.data['B'].isnull().values.any())
        
        # Test filling missing values with median
        self.dc.fill_missing_values('B', method='median')
        self.assertFalse(self.dc.data['B'].isnull().values.any())
        
        # Test filling missing values with mode
        self.dc.fill_missing_values('C', method='mode')
        self.assertFalse(self.dc.data['C'].isnull().values.any())
        
        # Test filling missing values with value
        self.dc.fill_missing_values('B', value=0, method='value')
        self.assertFalse(self.dc.data['B'].isnull().values.any())
    
    def test_drop_missing_values(self):
        # Test dropping missing values
        self.dc.drop_missing_values()
        self.assertEqual(len(self.dc.data), self.len_data-1)
    
    def test_convert_to_numeric(self):
        # Test converting a column to numeric
        self.dc.convert_to_numeric('B')
        self.assertTrue(pd.api.types.is_numeric_dtype(self.dc.data['B']))
    
    def test_encode_categorical(self):
        # Test encoding a categorical column
        self.dc.encode_categorical('C')
        self.assertTrue(pd.api.types.is_numeric_dtype(self.dc.data['C']))
        
if __name__ == '__main__':
    unittest.main()
