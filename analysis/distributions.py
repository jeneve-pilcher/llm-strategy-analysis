import pandas as pd
import numpy as np
from pathlib import Path

path = Path(__file__).resolve().parent.parent / 'data' # Go up to repo root then down to data folder

#{model: df}
def load(folder):
    return {csv.stem: pd.read_csv(csv) for csv in folder.glob("*.csv")}
product = load(path / 'product')
business = load(path / 'business')




product_dist = {}
for model, df in product.items():
    data = df.iloc[0:,1:].to_numpy()
    product_dist[model]=data

product_plot = {}
for model, data in product_dist.items():
    mean = np.mean(data)
    std = np.std(data)
    product_plot[model]=(mean,std)
print(product_plot)



#print(isinstance(business,pd.DataFrame))
#businessdf =pd.DataFrame(business)
#businessnums = business.iloc[1:,1:]

