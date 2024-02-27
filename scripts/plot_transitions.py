import numpy as np
import matplotlib.pyplot as plt

transitions = np.loadtxt("Results/transitions/out1.txt")
tmax = 50
time = [t for t in range(1,(tmax)+1)]
for pair in range(0, transitions.shape[0], tmax):
    fig, ax = plt.subplots()
    from_region = int(transitions[pair, 0])
    to_region = int(transitions[pair, 1])
    abm_kedaechtnislos = transitions[pair:(pair+tmax), 3]
    abm_real = transitions[pair:(pair+tmax), 4]
    pdmm = transitions[pair:(pair+tmax), 5]
    #smm = transitions[pair:(pair+tmax), 6]
    real = transitions[pair:(pair+tmax), 6]
    ax.plot(time, abm_kedaechtnislos, label='abm kedaechtnislos')
    ax.plot(time, abm_real, label='abm real')
    ax.plot(time, pdmm, label='pdmm')
    #ax.plot(time, smm, label='smm')
    ax.plot(time, real, label='real data')
    ax.legend()
    ax.set_xlabel('Time')
    ax.set_ylabel('#Transitions')
    ax.set_title('From ' + str(from_region) + ' to ' + str(to_region))
    fig.savefig(str(from_region) + "->" + str(to_region) + ".png")
    print()