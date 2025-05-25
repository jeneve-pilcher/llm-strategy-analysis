import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

path = Path(__file__).resolve().parent.parent / 'data' # Go up to repo root then down to data folder

#{model: df}
def load(folder):
    return {csv.stem: pd.read_csv(csv) for csv in folder.glob("*.csv")}
product = load(path / 'product')
business = load(path / 'business')


#model: cleaned df (rid of labels --> just numbers)
business_dist = {}
for model, df in business.items():
    data = df.iloc[0:,1:]
    business_dist[model]=data
    stats.probplot(data.values.flatten(), dist="norm", plot=plt)
    plt.title(f"Q-Q Plot: Normality Check for {model}")
    plt.grid(True)
    plt.savefig(f"./results/bus/distributions/is-normal/QQ/{model}.png")
    plt.show()

    sns.histplot(data.values.flatten(), kde=True)
    plt.title(f"Histogram with KDE for {model}")
    plt.savefig(f"./results/bus/distributions/is-normal/KDE/{model}.png")
    plt.show()

sns.kdeplot(business_dist['claude'].values.flatten(), label='Claude')
sns.kdeplot(business_dist['gpt'].values.flatten(), label='GPT')
sns.kdeplot(business_dist['deepseek'].values.flatten(), label='Deepseek')
sns.kdeplot(business_dist['gemini'].values.flatten(), label='Gemini')
plt.legend()
plt.title("Ranked Effectiveness of Business Strategies Created by LLMs")
plt.savefig(f"./results/bus/distributions/dub.png")
plt.show()

#{model:(mean,std)}
# product_plot = {}
# for model, data in business_dist.items():
#     product_nums = data.to_numpy()
#     mean = np.mean(product_nums)
#     std = np.std(product_nums)
#     product_plot[model]=(mean,std)



