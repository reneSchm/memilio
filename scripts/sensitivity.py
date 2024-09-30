import pandas as pd
import matplotlib.pyplot as plt
path = "cpp/outputs/sensitivity_analysis/20240930_v1/Hybrid_elem_effects0.txt"
df = pd.read_csv(path, sep=" ").drop(columns=['Unnamed: 10'])
labels = ['C', 'E', 'I', 'r', r'$\mu_C^R$', r'$\mu_I^D$', r'$\sigma$', r'$\tau_E$', r'$\lambda_i^{(k,l)}$', r'$\rho$']
fig, axs = plt.subplots(1, 2, figsize=(10,4))
axs[0].boxplot(df, labels=labels)
axs[1].barh(labels, [df[col].mean() for col in df.columns])
axs[1].set_xscale("symlog")
fig.suptitle('Norm Infected')
fig.savefig("scripts/Results/sensitivity_analysis/20240930_v1/Hybrid_boxplot_norm_infected.png")
