import pandas as pd
import matplotlib.pyplot as plt

label_map = {'C': r'$C_0$', 'E': r'$E_0$', 'I': r'$I_0$', 'contact_radius': r'$r$', 'mu_C_R': r'$\mu_C^R$', 'mu_I_D': r'$\mu_I^D$',
             'sigma': r'$\sigma$', 't_Exposed': r'$\tau_E$', 't_Carrier': r'$\tau_C$', 't_Infected': r'$\tau_I$',
             'transition_rates': r'$\lambda_i^{(k,l)}$', 'transmission_rate': r'$\rho$'}

"""
Boxplot of distribution and (log-scaled) bar plot of mean.
@param output_folders Folders with output files i.e. if there are runs on multiple nodes, the results are stored in seperate folders
@param output_file_name Name of output files e.g. ABM_elem_effects
@param titles of different output values e.g. Norm Imfected, Total Deaths etc.
@param saving_path path to save the png
"""
def plot_results(output_folders, output_file_name, titles, saving_path):
    fig, axs = plt.subplots(len(titles), 2, figsize=(12,17), constrained_layout=True)
    for i in range(len(titles)):
        path = output_folders[0] + output_file_name + str(i) + ".txt"
        df = pd.read_csv(path, sep=" ").drop(columns=['Unnamed: 10'])
        for f in range(1, len(output_folders)):
            path = output_folders[f] + output_file_name + str(i) + ".txt"
            df = pd.concat([df, pd.read_csv(path, sep=" ").drop(columns=['Unnamed: 10'])], ignore_index=True)
        #get labels
        labels = []
        for c in df.columns:
            labels.append(label_map[c])
        axs[i, 0].boxplot(df, labels=labels)
        axs[i, 1].barh(labels, [df[col].mean() for col in df.columns])
        axs[i, 1].set_xscale("symlog")
        axs[i, 0].set_title(titles[i])
    fig.savefig(saving_path + output_file_name + ".png")

plot_results(['cpp/outputs/sensitivity_analysis/20240930_v1/'], 'Hybrid_elem_effects', 
             ['Norm Infected', 'Max Infected', 'Total Transmissions', 'Total Deaths'], "scripts/Results/sensitivity_analysis/20240930_v1/")
