import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#from scipy.stats import f_oneway
#from statsmodels.stats.multicomp import pairwise_tukeyhsd
from distributions import product_dict

# --------------- Prepare data ---------------
# Flatten and collect all scores + group labels
# data = []
# labels = []


# for model, values in product_dict.items():
#     values = np.array(values).flatten()
#     data.extend(values)
#     labels.extend([model] * len(values))

# df = pd.DataFrame({'score': data, 'model': labels})

# # --------------- Run ANOVA ---------------

# stat, p = f_oneway(
#     product_dist['gpt'].to_numpy().flatten(),
#     product_dist['deepseek'].to_numpy().flatten(),
#     product_dist['claude'].to_numpy().flatten(),
#     product_dist['gemini'].to_numpy().flatten()
# )

# print(f"ANOVA F-statistic: {stat:.4f}, p-value: {p:.4e}")


# # --------------- Tukey’s HSD Test ---------------
# tukey = pairwise_tukeyhsd(endog=df['score'], groups=df['model'], alpha=0.05)
# print(tukey)

# # --------------- Plot 1: Tukey HSD Visualization ---------------
# fig1 = tukey.plot_simultaneous(comparison_name='gpt', ylabel='Model')
# plt.title("Tukey HSD: Pairwise Differences (95% CI)")
# plt.grid(True)
# plt.tight_layout()
# plt.show()

# --------------- Plot 2: KDE + Mean/Std Overlay ---------------
plt.figure(figsize=(10, 6))

palette = sns.color_palette("tab10", len(product_dict))

for i, (model, values) in enumerate(product_dict.items()):
    values = np.array(values).flatten()
    sns.kdeplot(values, label=model, fill=False, alpha=1.0, color=palette[i])
    
    mean = np.mean(values)
    std = np.std(values)

    # Mean line
    plt.axvline(mean, color=palette[i], linestyle='--', linewidth=2, label = None)

    # Std lines
    # Vertical ticks at ±1 std
    plt.plot([mean - std, mean - std], [0, 0.02], color=palette[i], linewidth=2)
    plt.plot([mean + std, mean + std], [0, 0.02], color=palette[i], linewidth=2)
    # plt.axvline(mean + std, color=palette[i], linestyle='--', alpha=0.6)
    # plt.axvline(mean - std, color=palette[i], linestyle='--', alpha=0.6)

plt.title("Peer-and-Self-Ranked Effectiveness of Product Strategies Created by LLMs")
plt.xlabel("Score out of 10")
plt.ylabel("Density: Rate that the Score Occurred")
plt.legend()
plt.tight_layout()
plt.savefig(f"./results/bus/distributions/finalproddis.png")
plt.show()
