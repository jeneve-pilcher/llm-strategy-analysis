import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from distributions import business_dist

# --------------- Prepare data ---------------
# Flatten and collect all scores + group labels
data = []
labels = []


for model, values in business_dist.items():
    values = np.array(values).flatten()
    data.extend(values)
    labels.extend([model] * len(values))

df = pd.DataFrame({'score': data, 'model': labels})

# --------------- Run ANOVA ---------------

stat, p = f_oneway(
    business_dist['gpt'].to_numpy().flatten(),
    business_dist['deepseek'].to_numpy().flatten(),
    business_dist['claude'].to_numpy().flatten(),
    business_dist['gemini'].to_numpy().flatten()
)

print(f"ANOVA F-statistic: {stat:.4f}, p-value: {p:.4e}")


# --------------- Tukey’s HSD Test ---------------
tukey = pairwise_tukeyhsd(endog=df['score'], groups=df['model'], alpha=0.05)
print(tukey)

# --------------- Plot 1: Tukey HSD Visualization ---------------
fig1 = tukey.plot_simultaneous(comparison_name='gpt', ylabel='Model')
plt.title("Tukey HSD: Pairwise Differences (95% CI)")
plt.grid(True)
plt.tight_layout()
plt.show()

# --------------- Plot 2: KDE + Mean/Std Overlay ---------------
plt.figure(figsize=(10, 6))

palette = sns.color_palette("tab10", len(business_dist))

for i, (model, values) in enumerate(business_dist.items()):
    values = np.array(values).flatten()
    sns.kdeplot(values, label=model, fill=False, alpha=0.3, color=palette[i])
    
    mean = np.mean(values)
    std = np.std(values)

    # Mean line
    plt.axvline(mean, color=palette[i], linestyle='-', linewidth=2, label=f"{model} mean")

    # Std lines
    plt.axvline(mean + std, color=palette[i], linestyle='--', alpha=0.6)
    plt.axvline(mean - std, color=palette[i], linestyle='--', alpha=0.6)

plt.title("Distributions with Means and ±1 Std Dev")
plt.xlabel("Score")
plt.legend()
plt.tight_layout()
plt.show()
