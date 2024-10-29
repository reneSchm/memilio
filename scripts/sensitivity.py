import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

total_label_map = {'C': r'$C_0$', 'E': r'$E_0$', 'I': r'$I_0$', 'contact_radius': r'$r$', 'mu_C_R': r'$\mu_C^R$', 'mu_I_D': r'$\mu_I^D$',
             'sigma': r'$\sigma$', 't_Exposed': r'$\tau_E$', 't_Carrier': r'$\tau_C$', 't_Infected': r'$\tau_I$',
             'transition_rates': r'$\lambda_i^{(k,l)}$', 'transmission_rate': r'$\rho$', 'commute_weights': r'$\lambda_i^{(k,l)}$', 
             'dummy': 'dummy', 'dummy1': 'dummy', 'dummy2': 'dummy'}

ABM_label_map = {'C': r'$C_0$', 'E': r'$E_0$', 'I': r'$I_0$', 'contact_radius': r'$r$', 'mu_C_R': r'$\mu_C^R$', 'mu_I_D': r'$\mu_I^D$',
             'sigma': r'$\sigma$', 't_Exposed': r'$\tau_E$', 't_Carrier': r'$\tau_C$', 't_Infected': r'$\tau_I$',
             'transition_rates': "dummy", 'transmission_rate': r'$\rho$', 'commute_weights': r'$\lambda_i^{(k,l)}$',
             'dummy': 'dummy', 'dummy1': 'dummy', 'dummy2': 'dummy'}

PDMM_label_map = {'C': r'$C_0$', 'E': r'$E_0$', 'I': r'$I_0$', 'contact_radius': "dummy", 'mu_C_R': r'$\mu_C^R$', 'mu_I_D': r'$\mu_I^D$',
             'sigma': "dummy", 't_Exposed': r'$\tau_E$', 't_Carrier': r'$\tau_C$', 't_Infected': r'$\tau_I$',
             'transition_rates': r'$\lambda_i^{(k,l)}$', 'transmission_rate': r'$\rho$', 'commute_weights': r'$\lambda_i^{(k,l)}$',
             'dummy': 'dummy', 'dummy1': 'dummy', 'dummy2': 'dummy'}

font_size = 16
plt.rc ('font', size = font_size) # steuert die Standardtextgröße
plt.rc ('axes', titlesize = font_size) # Schriftgröße des Titels
plt.rc ('axes', labelsize = font_size) # Schriftgröße der x- und y-Beschriftungen
plt.rc ('xtick', labelsize = font_size) #Schriftgröße der x-Tick-Labels
plt.rc ('ytick', labelsize = font_size) #Schriftgröße der y-Tick-Labels
plt.rc ('legend', fontsize = font_size) #Schriftgröße der Legende

"""
Boxplot of distribution and (log-scaled) bar plot of mean.
@param output_folders Folders with output files i.e. if there are runs on multiple nodes, the results are stored in seperate folders
@param output_file_name Name of output files e.g. ABM_elem_effects
@param titles of different output values e.g. Norm Imfected, Total Deaths etc.
@param saving_path path to save the png
"""
def plot_results(output_folders, output_file_name, titles, saving_path, label_map, fig_size = (12,17), display_dummies = True):
    fig, axs = plt.subplots(len(titles), 2, figsize=fig_size, constrained_layout=True)
    if(titles[0] == 'Time'):
        path = output_folders[0] + output_file_name + str(4) + ".txt"
        df = pd.read_csv(path, sep=" ")
        df = df.drop(columns=['Unnamed: ' + str(len(df.columns)-1)])
        for f in range(1, len(output_folders)):
            path = output_folders[f] + output_file_name + str(4) + ".txt"
            df = pd.concat([df, pd.read_csv(path, sep=" ").drop(columns=['Unnamed: 12'])], ignore_index=True)
        #get labels
        labels = []
        width = 0.25
        dummy_lower = df.max().max()
        dummy_upper = df.min().min()
        for c in df.columns:
            labels.append(label_map[c])
            if(label_map[c] == "dummy"):
                dummy_lower = min(np.percentile(np.array(df[c]), 25), dummy_lower)
                dummy_upper = max(np.percentile(np.array(df[c]), 75), dummy_upper)
        axs[0].boxplot(df, labels=labels, showmeans=True)
        axs[0].axhspan(dummy_lower, dummy_upper, facecolor="gray", alpha=0.3)
        axs[0].set_yscale("symlog")
        axs[1].barh(np.arange(len(labels)), [df[col].mean() for col in df.columns], label="mean", height=0.5)
        axs[1].barh(np.arange(len(labels)) + width, [df[col].median() for col in df.columns], label="median", height=0.5)
        #axs 1].barh(np.arange(len(labels)) + width, [df[col].std() for col in df.columns], label="std", height=0.5)
        axs[1].set_yticks(np.arange(len(labels))+width, labels)
        axs[1].set_xscale("symlog")
        axs[1].legend()
        axs[0].set_title(titles[0])
    else:    
        for i in range(len(titles)):
            path = output_folders[0] + output_file_name + str(i+1) + ".txt"
            df = pd.read_csv(path, sep=" ")
            df = df.drop(columns=['Unnamed: ' + str(len(df.columns)-1)])
            for f in range(1, len(output_folders)):
                path = output_folders[f] + output_file_name + str(i+1) + ".txt"
                df = pd.concat([df, pd.read_csv(path, sep=" ").drop(columns=['Unnamed: 13'])], ignore_index=True)
            #get labels
            labels = []
            width = 0.25
            dummy_lower = df.max().max()
            dummy_upper = df.min().min()
            for c in df.columns:
                labels.append(label_map[c])
                if(label_map[c] == "dummy"):
                    if((c != "dummy1")):
                        dummy_lower = min(np.percentile(np.array(df[c]), 5), dummy_lower)
                        dummy_upper = max(np.percentile(np.array(df[c]), 95), dummy_upper)
                    if(not display_dummies):
                        labels.remove(label_map[c])
                        df = df.drop(columns=[c])
            axs[i, 0].boxplot(df, labels=labels, showmeans=True)
            axs[i, 0].set_yscale("symlog")
            axs[i, 1].barh(np.arange(len(labels)), [df[col].mean() for col in df.columns], label="Mean", height=0.5)
            axs[i, 1].barh(np.arange(len(labels)) + width, [df[col].median() for col in df.columns], label="Median", height=0.5)
            #axs[i, 1].barh(np.arange(len(labels)) + width, [df[col].std() for col in df.columns], label="std", height=0.5)
            axs[i, 1].set_yticks(np.arange(len(labels))+width, labels)
            axs[i, 1].set_xscale("symlog")
            axs[i, 1].legend(loc='upper left')
            axs[i, 0].set_title(titles[i])
            axs[i, 0].axhspan(dummy_lower, dummy_upper, facecolor="gray", alpha=0.3)
    fig.savefig(saving_path + output_file_name + ".png")

# REL EFFECTS
# Values
plot_results(['cpp/outputs/20241025_Munich_hybrid_final1/'], 'Hybrid_rel_effects', 
             [r'$\max_{t}N_{I}(t)$', r'$\sum_{t=0}^{t_{max}}(S\rightarrow E)(t)$', r'$\sum_{t=0}^{t_{max}}N_{D}(t)$'], "scripts/Results/20241025_Munich_hybrid_final1/", total_label_map, display_dummies=False, fig_size=(14, 15))
# DIFFS
# Values
plot_results(['cpp/outputs/20241025_Munich_hybrid_final1/'], 'Hybrid_diff', 
            ['Max Infected', 'Total Transmissions', 'Total Deaths'], "scripts/Results/20241025_Munich_hybrid_final1/", total_label_map, display_dummies=False, fig_size=(14, 15))

#[r'$\max_{t}N_{I}(t)$', r'$\sum_{t=0}^{t_{max}}(S\rightarrow E)(t)$', r'$\sum_{t=0}^{t_{max}}N_{D}(t)$']
