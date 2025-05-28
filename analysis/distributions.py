import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from scipy.stats import f_oneway

path = Path(__file__).resolve().parent.parent / 'data' # Go up to repo root then down to data folder

#{model: df}
def load(folder):
    return {csv.stem: pd.read_csv(csv) for csv in folder.glob("*.csv")}
product = load(path / 'product')
business = load(path / 'business')


#model: cleaned df (rid of labels --> just numbers)
business_dict = {}
for model, df in business.items():
    data = df.iloc[0:,1:]
    business_dict[model]=data

product_dict = {}
for model, df in product.items():
    data = df.iloc[0:,1:]
    product_dict[model]=data



##ANNOVA need to learn?
# stat, p = f_oneway(business_dist["gpt"], business_dist["deepseek"], business_dist["claude"], business_dist["gemini"])
# print(f"ANOVA: p = {p}")

# pvals = np.array([2.86862245e-02, 2.10708689e-02, 9.73643134e-01, 3.21571280e-04,
#                   6.08766871e-02, 8.07468282e-04, 8.09245873e-02, 1.86626146e-02,
#                   2.99908149e-03, 5.20798018e-01])

# corrected_alpha = 0.05 / len(pvals)
# significant = pvals < corrected_alpha

# print("Corrected alpha:", corrected_alpha)
# print("Significant tests:", significant)


## QQ for individ. model
#     stats.probplot(data.values.flatten(), dist="norm", plot=plt)
#     plt.title(f"Q-Q Plot: Normality Check for {model}")
#     plt.grid(True)
#     plt.savefig(f"./results/bus/distributions/is-normal/QQ/{model}.png")
#     plt.show()


##make KDE for individ. model
#     sns.histplot(data.values.flatten(), kde=True)
#     plt.title(f"Histogram with KDE for {model}")
#     plt.savefig(f"./results/bus/distributions/is-normal/KDE/{model}.png")
#     plt.show()


#KDE all model no mean/std
# sns.kdeplot(business_dist['claude'].values.flatten(), label='Claude')
# sns.kdeplot(business_dist['gpt'].values.flatten(), label='GPT')
# sns.kdeplot(business_dist['deepseek'].values.flatten(), label='Deepseek')
# sns.kdeplot(business_dist['gemini'].values.flatten(), label='Gemini')
# plt.legend()
# plt.title("Peer-and-Self-Ranked Effectiveness of Business Strategies Created by LLMs")
# plt.savefig(f"./results/bus/distributions/dub.png")
# plt.show()


##calculate mean/std --> {model:(mean,std)}
# product_plot = {}
# for model, data in business_dist.items():
#     product_nums = data.to_numpy()
#     mean = np.mean(product_nums)
#     std = np.std(product_nums)
#     product_plot[model]=(mean,std)



