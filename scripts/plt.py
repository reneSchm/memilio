import numpy as np
import matplotlib.pyplot as plt

def get_lists():
    with open("output.txt") as file:
        s = file.readlines()
        i1 = s.index("Global Model 1/2\n")
        i2 = s.index("Global Model 2/2\n")
    return np.loadtxt(s[i1 : i2], skiprows=2), np.loadtxt(s[i2 : ], skiprows=2)

for l in get_lists():
    x = l[:, 0]
    ys = [np.zeros_like(x)]
    for y in reversed([l[:,i] for i in range(1, len(l[0]))]):
        ys = [y + ys[0]] + ys
    #    ys += [ys[i-1] + l[i, :]]
    labels = "SIR"
    
    plt.figure()
    for i in range(len(ys) - 1):
        plt.fill_between(x, ys[i], ys[i+1], label=labels[i])
        plt.plot(x, ys[i], c='black')
    plt.plot(x, ys[-1], c='black')
    plt.legend()
plt.show()

