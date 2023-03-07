import pandas as pd
from sqlalchemy import create_engine
import json

class DataImporter:
    '''
        DataImporter class with 
        __init__ method that takes a filename as an argument: This filename is used by the various import methods to load the data.
    '''
    def __init__(self, filename):
        self.filename = filename

    def import_csv(self, **kwargs):
        '''
        The import_csv method reads a CSV file 
        using pd.read_csv and returns a pandas DataFrame. 
        It accepts optional keyword arguments that are passed to pd.read_csv.

        Ex: 
        
        import data_import

        filename = 'data.csv'
        data = data_import.DataImport(filename)
        df = data.import_csv(delimiter=';')
        '''
        return pd.read_csv(self.filename, **kwargs)

    def import_excel(self, sheet_name=0, **kwargs):
        return pd.read_excel(self.filename, sheet_name=sheet_name, **kwargs)

    def import_sql(self, url, query):
        engine = create_engine(url)
        return pd.read_sql(query, engine)

    def import_json(self):
        with open(self.filename, 'r') as f:
            data = json.load(f)
        return pd.json_normalize(data)
    
    def import_pickle(self):
        return pd.read_pickle(self.filename)
