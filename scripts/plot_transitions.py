import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

p05 = np.loadtxt("../cpp/0.05_num_transitions.txt")
p25 = np.loadtxt("../cpp/0.25_num_transitions.txt")
p50 = np.loadtxt("../cpp/0.50_num_transitions.txt")
p75 = np.loadtxt("../cpp/0.75_num_transitions.txt")
p95 = np.loadtxt("../cpp/0.95_num_transitions.txt")

font_size = 16

plt.rc ('font', size = font_size) # steuert die Standardtextgröße
plt.rc ('axes', titlesize = font_size) # Schriftgröße des Titels
plt.rc ('axes', labelsize = font_size) # Schriftgröße der x- und y-Beschriftungen
plt.rc ('xtick', labelsize = font_size) #Schriftgröße der x-Tick-Labels
plt.rc ('ytick', labelsize = font_size) #Schriftgröße der y-Tick-Labels
plt.rc ('legend', fontsize = font_size) #Schriftgröße der Legende

tmax = 50
time = [t for t in range(1,(tmax)+1)]
# for pair in range(0, transitions.shape[0], tmax):
#     fig, ax = plt.subplots()
#     from_region = int(transitions[pair, 0])
#     to_region = int(transitions[pair, 1])
#     abm_kedaechtnislos = transitions[pair:(pair+tmax), 3]
#     abm_real = transitions[pair:(pair+tmax), 4]
#     pdmm = transitions[pair:(pair+tmax), 5]
#     #smm = transitions[pair:(pair+tmax), 6]
#     real = transitions[pair:(pair+tmax), 6]
#     ax.plot(time, abm_kedaechtnislos, label='abm kedaechtnislos')
#     ax.plot(time, abm_real, label='abm real')
#     ax.plot(time, pdmm, label='pdmm')
#     #ax.plot(time, smm, label='smm')
#     ax.plot(time, real, label='real data')
#     ax.legend()
#     ax.set_xlabel('Time')
#     ax.set_ylabel('#Transitions')
#     ax.set_title('From ' + str(from_region) + ' to ' + str(to_region))
#     fig.savefig(str(from_region) + "->" + str(to_region) + ".png")
#     print()
fig_from, ax_from = plt.subplots(2, 4, figsize=(20, 8))
#fig_from.set_size_inches(20, 10, forward=True)
plt.figtext(0.45, 0.025, "Time(Days)", size=font_size+2)
plt.figtext(0.075, 0.3, "Number of Transitions", size=font_size+2, rotation='vertical')
#fig_from.text(0, 0, 'test', fontsize=30)
#fig_from, ax_from = plt.subplots()
# fig_to, ax_to = plt.subplots()
# plt.xlabel("Time(Days)")
# plt.ylabel("Number of daily transitions")
# ax_from[0, 0].set_title("Mobility from Munich to surrounding counties")
# ax_to.set_xlabel("Time[Days]")
# ax_to.set_ylabel("#Transitions")
# ax_to.set_title("Transitions from surrounding regions to Munich")

labels = {0: "Fürstenfeldbruck", 1: "Dachau", 2: "Starnberg", 3: "München", 4: "München Land", 
 5: "Freising", 6: "Erding", 7: "Ebersberg"}
#colors = {0: "rosybrown", 1: "chocolate", 2: "teal", 3: "blueviolet", 4: "darkblue", 5: "grey", 6: "forestgreen", 7: "orange"}
colors = {0: "tab:blue", 1: "tab:orange", 2: "tab:green", 3: "tab:red", 4: "tab:purple", 5: "tab:brown", 6: "tab:pink", 7: "tab:gray"}
for pair in range(0, p50.shape[0], tmax):
    from_region = int(p50[pair, 0])
    to_region = int(p50[pair, 1])
    real = p50[pair:(pair+tmax), 5]
    abm_p05 = p05[pair:(pair+tmax), 3]
    abm_p95 = p95[pair:(pair+tmax), 3]
    if(from_region == 3):
        if(to_region < 4):
            i = 0
            j = to_region
        else:
            i = 1
            j = to_region - 4
        #ax_from.plot(time, abm_p95, color=colors[to_region], alpha=0.2)
        ax_from[i, j].fill_between(time, abm_p05, abm_p95, color=colors[to_region], alpha=0.3)
        ax_from[i, j].plot(time, p50[pair:(pair+tmax), 3], color=colors[to_region], alpha=0.4, linewidth=3) #label = labels[to_region]
        ax_from[i, j].plot(time, real, linestyle="dotted", color=colors[to_region], linewidth=3) #label = "_static register data"
        ax_from[i, j].set_yticks(np.arange(round(min(abm_p05)-10, -1), round(min(abm_p95)+50, -1), step=20)) #np.linspace(round(min(abm_p05)-5, -1), round(max(abm_p95)+10, -1), 5, dtype=int)[1:-1]
    # if(to_region == 3):
    #     ax_to.plot(time, abm_real, color=colors[from_region], label = "ABM " + labels[to_region])
    #     ax_to.plot(time, real, "--", color=colors[from_region], label = "static register data")
# legend_to = ax_to.legend(loc='center left', bbox_to_anchor=(1, 0.5))
r0 = Line2D([0], [0], label="Fürstenfeldbruck", color=colors[0])
r1 = Line2D([0], [0], label="Dachau", color=colors[1])
r2 = Line2D([0], [0], label="Starnberg", color=colors[2])
r4 = Line2D([0], [0], label="München Land", color=colors[4])
r5 = Line2D([0], [0], label="Freising", color=colors[5])
r6 = Line2D([0], [0], label="Erding", color=colors[6])
r7 = Line2D([0], [0], label="Ebersberg", color=colors[7])
r_real = Line2D([0], [0], label="Scaled static register data", color="black", linestyle="--")

handles, labels = ax_from[0, 3].get_legend_handles_labels()
handles.extend([r0, r1, r2, r4, r5, r6, r7, r_real])

ax_from[0, 3].set_axis_off()
legend_from = ax_from[0, 3].legend(handles=handles,loc='center', bbox_to_anchor=(0.4, 0.5)) #,loc='center left', bbox_to_anchor=(1, 0.5)
#plt.yscale("log")
fig_from.savefig("MunichTransitions.png", bbox_extra_artists=(legend_from,))
# fig_to.savefig("TransitionsToMunich.png", bbox_extra_artists=(legend_to,), bbox_inches='tight')
