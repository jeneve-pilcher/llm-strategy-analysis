import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

path = Path(__file__).resolve().parent.parent / 'data' # Go up to repo root then down to data folder

class DistributionDataLoader:
    """
    A class to load and process distribution data from CSV files.
    """

    def __init__(self, path=path):
        """
        Initializes the DistributionDataLoader with a path to the data folder.
        :param path: Path to the folder containing CSV files.
        """
        self._path = path

    def load_folder(self,folder):
        """
        Loads all CSV files in a given folder into a dictionary of DataFrames.
        :param folder: Path to the folder containing CSV files.
        :return: Dictionary where keys are filenames (without extension) and values are DataFrames.
        """
        return {csv.stem: pd.read_csv(csv) for csv in folder.glob("*.csv")}
    
    def load(self):
        """
        Loads product and business data from their respective folders.
        :return: Tuple of dictionaries containing product and business data.
        """
        product = self.load_folder(self._path / 'product')
        business = self.load_folder(self._path / 'business')
        
        business_dict = {}
        for model, df in business.items():
            data = df.iloc[0:,1:]
            business_dict[model]=data

        product_dict = {}
        for model, df in product.items():
            data = df.iloc[0:,1:]
            product_dict[model]=data

        return product_dict, business_dict













