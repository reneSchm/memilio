import matplotlib.pyplot as plt
import pandas as pd

font_size = 16

plt.rc ('font', size = font_size) # steuert die Standardtextgröße
plt.rc ('axes', titlesize = font_size) # Schriftgröße des Titels
plt.rc ('axes', labelsize = font_size) # Schriftgröße der x- und y-Beschriftungen
plt.rc ('xtick', labelsize = font_size) #Schriftgröße der x-Tick-Labels
plt.rc ('ytick', labelsize = font_size) #Schriftgröße der y-Tick-Labels
plt.rc ('legend', fontsize = font_size) #Schriftgröße der Legende

#na_list = [80, 400, 800, 1000, 4000, 8000, 12000, 16000, 20000, 30000, 40000]
na_list = [699, 1402, 2809, 5624, 14064, 28133, 40190]
model_list = ["ABM", "PDMM", "Hybrid_comps"]
colors = {'ABM': 'indianred', 'PDMM': 'royalblue', 'Hybrid_comps': 'green'}
labels = {'ABM': 'ABM', 'PDMM': 'PDMM', 'Hybrid_comps': 'Spatial Hybrid'}
markers = {'ABM': 's', 'PDMM': 'x', 'Hybrid_comps': 'o'}

dir = "cpp/outputs/Munich_scaling/"
save_dir = "scripts/Results/Munich_scaling/"

fig, ax = plt.subplots(figsize = (8,6))
for model in model_list:
    x = []
    mean = []
    p05 = []
    p95 = []
    for na in na_list:
        x.append(na)
        mean.append(pd.read_csv(dir + str(na) + model + "_output_mean.txt", sep = " ").loc[0][1])
        p05.append(pd.read_csv(dir + str(na) + model + "_output_p05.txt", sep = " ").loc[0][1])
        p95.append(pd.read_csv(dir + str(na) + model + "_output_p95.txt", sep = " ").loc[0][1])
    ax.plot(x, mean, color = colors[model], label = labels[model], marker = markers[model])
    # ax.plot(x, p05, color = colors[model], linestyle = "dotted", alpha=0.3)
    # ax.plot(x, p95, color = colors[model], linestyle = "dotted", alpha=0.3)
    # ax.fill_between(x, p05, p95, alpha=0.2, color=colors[model])
plt.legend(bbox_to_anchor=(0.5, 1.08), loc="center", ncol = 3)
plt.grid()
ax.set_xlabel('Agents[#]')
ax.set_ylabel('Time[s]')
plt.yscale("log")
plt.xscale("log")
plt.tight_layout()
fig.subplots_adjust(top=0.89)
fig.savefig(save_dir +"scaling.png")
