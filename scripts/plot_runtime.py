import matplotlib.pyplot as plt

#num agents
x = [80, 400, 800, 1000, 4000, 8000, 12000, 16000, 20000, 30000, 40000]

#times
ABM = [0.113394, 1.66601, 5.700355, 8.32829, 327.9895, 2367.125, 7024.37, 16417.4, 31201.5, 102952, 244838]
PDMM = [0.00287508, 0.01307685, 0.02213445, 0.02614235, 0.08054100, 0.15327950, 0.22151100, 0.29598100, 0.36933050, 0.54956850, 0.73771900]
Hybrid = [0.04858460, 0.30229750, 0.87938, 1.498875, 15.1982, 60.8983, 160.1410, 335.6770, 588.1375, 1430.02, 3593.55]

fig, ax = plt.subplots()
ax.plot(x, ABM, marker='s', label='ABM', color='indianred')
ax.plot(x, Hybrid, marker='o', label='Hybrid', color='royalblue')
ax.plot(x, PDMM, marker='x', label='PDMM', color='green')
ax.set_xlabel('Number of Agents')
ax.set_ylabel('Time(seconds)')
ax.legend()
plt.yscale("log")
plt.xscale("log")
ax.grid(True, linestyle=':')
fig.savefig("time.png")