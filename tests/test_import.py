import sys
# import your modules and packages from anywhere, i.e., from any directory on your computer.
sys.path.append('../utils/utils')

import unittest
import pandas as pd
import data_importer

class TestDataImport(unittest.TestCase):
    '''
        In this test, we import the unittest module and the DataImport class from the data_import module. 
        We then define a TestDataImport class that inherits from unittest.TestCase. 
        In the setUp method of this class, we define the filenames and SQL query/URL that we will use for testing, 
        and create an instance of the DataImport class with the CSV file.

        We then define four test methods, 
        one for each of the import_csv, import_excel, import_json, and import_sql methods in the DataImport class. 
        In each of these methods, we call the corresponding method on the data_import object, and use the assertIsInstance method 
        from unittest to check that the result is a pandas.DataFrame.
    '''
    def setUp(self):
        self.csv_file = 'data.csv'
        self.excel_file = 'data.xlsx'
        self.json_file = 'data.json'
        self.sql_url = 'mysql://user:password@localhost/test_db'
        self.sql_query = 'SELECT * FROM data_table'
        self.data_import = data_importer.DataImport(self.csv_file)
        self.data_import.filename = 'data.pkl'

    def test_import_csv(self):
        df = self.data_import.import_csv()
        self.assertIsInstance(df, pd.DataFrame)

    def test_import_excel(self):
        self.data_import.filename = self.excel_file
        df = self.data_import.import_excel()
        self.assertIsInstance(df, pd.DataFrame)

    def test_import_json(self):
        self.data_import.filename = self.json_file
        df = self.data_import.import_json()
        self.assertIsInstance(df, pd.DataFrame)

    def test_import_sql(self):
        df = self.data_import.import_sql(url=self.sql_url, query=self.sql_query)
        self.assertIsInstance(df, pd.DataFrame)
        
    def test_import_pickle(self):
        df = self.data_import.import_pickle()
        self.assertIsInstance(df, pd.DataFrame)

if __name__ == '__main__':
    unittest.main()