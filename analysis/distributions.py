import pandas as pd
import numpy as np
from pathlib import Path

path = Path(__file__).resolve().parent.parent / 'data' # Go up to repo root then down to data folder

#{model name: df}
def load(folder):
    return {csv.stem: pd.read_csv(csv) for csv in folder.glob("*.csv")}

product = load(path / 'product')
business = load(path / 'business')

print(isinstance(product,pd.DataFrame))
productdf =pd.DataFrame(product)
productnums = product.iloc[1:,1:]

print(isinstance(business,pd.DataFrame))
businessdf =pd.DataFrame(business)
businessnums = business.iloc[1:,1:]

