import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

transitions = np.loadtxt("Results/transitions/num_trans.txt")
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

fig_from, ax_from = plt.subplots()
# fig_to, ax_to = plt.subplots()
ax_from.set_xlabel("Time(Days)")
ax_from.set_ylabel("Number of daily transitions")
ax_from.set_title("Mobility from Munich to surrounding counties")
# ax_to.set_xlabel("Time[Days]")
# ax_to.set_ylabel("#Transitions")
# ax_to.set_title("Transitions from surrounding regions to Munich")

labels = {0: "Fürstenfeldbruck", 1: "Dachau", 2: "Starnberg", 3: "München", 4: "München Land", 
 5: "Freising", 6: "Erding", 7: "Ebersberg"}
#colors = {0: "rosybrown", 1: "chocolate", 2: "teal", 3: "blueviolet", 4: "darkblue", 5: "grey", 6: "forestgreen", 7: "orange"}
colors = {0: "tab:blue", 1: "tab:orange", 2: "tab:green", 3: "tab:red", 4: "tab:purple", 5: "tab:brown", 6: "tab:pink", 7: "tab:gray"}
for pair in range(0, transitions.shape[0], tmax):
    from_region = int(transitions[pair, 0])
    to_region = int(transitions[pair, 1])
    abm_real = transitions[pair:(pair+tmax), 4]
    real = transitions[pair:(pair+tmax), 6]
    if(from_region == 3):
        ax_from.plot(time, abm_real, color=colors[to_region]) #label = labels[to_region]
        ax_from.plot(time, real, "--", color=colors[to_region]) #label = "_static register data"
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

handles, labels = ax_from.get_legend_handles_labels()
handles.extend([r0, r1, r2, r4, r5, r6, r7, r_real])

legend_from = ax_from.legend(handles=handles) #,loc='center left', bbox_to_anchor=(1, 0.5)
fig_from.savefig("MunichTransitions.png", bbox_extra_artists=(legend_from,), bbox_inches='tight')
# fig_to.savefig("TransitionsToMunich.png", bbox_extra_artists=(legend_to,), bbox_inches='tight')
