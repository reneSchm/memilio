import matplotlib.pyplot as plt

font_size = 16

plt.rc ('font', size = font_size) # steuert die Standardtextgröße
plt.rc ('axes', titlesize = font_size) # Schriftgröße des Titels
plt.rc ('axes', labelsize = font_size) # Schriftgröße der x- und y-Beschriftungen
plt.rc ('xtick', labelsize = font_size) #Schriftgröße der x-Tick-Labels
plt.rc ('ytick', labelsize = font_size) #Schriftgröße der y-Tick-Labels
plt.rc ('legend', fontsize = font_size) #Schriftgröße der Legende

#num agents

###QUADWELL###
# x = [80, 400, 800, 1000, 4000, 8000, 12000, 16000, 20000, 30000, 40000]

###MUNICH###
x = [51, 276, 558, 2809, 5624, 14064, 28133, 56268, 70337, 93785]

#times

###QUADWELL###
# Teamserver
# ABM = [0.113394, 1.66601, 5.700355, 8.32829, 327.9895, 2367.125, 7024.37, 16417.4, 31201.5, 102952, 238537]
# PDMM = [0.00287508, 0.01307685, 0.02213445, 0.02614235, 0.08054100, 0.15327950, 0.22151100, 0.29598100, 0.36933050, 0.54956850, 0.73771900]
# Hybrid = [0.04858460, 0.30229750, 0.87938, 1.498875, 15.1982, 60.8983, 160.1410, 335.6770, 588.1375, 1430.02, 3593.55]

# Cluster
# ABM = [0.10114, 1.27346, 5.91025, 9.99055, 377.909, 2709.4, 8335.42, 19363.1, 37584.3, 128794, 308476]
# PDMM = [0.0054624, 0.0167232, 0.0268689, 0.0335557, 0.0822669, 0.13755, 0.194919, 0.275868, 0.339651, 0.462916, 0.626926]
# Hybrid = [0.0548094, 0.229091, 0.624175, 1.03139, 15.1724, 68.1663, 170.408, 337.172, 569.439, 1623.72, 3412.97]

###MUNICH###
# Teamserver
# ABM = [0.29, 0.68, 1.34, 18.0, 67.0, 511.33, 4908.33, 26317.33, 108085]
# PDMM = [0.018, 0.097, 0.199, 0.999, 2.0, 5.14, 10.05, 20.1, 39.4899]
# Hybrid = [0.33, 0.82, 1.49, 10.3, 32.5, 160.66, 1151, 7844.66, 20296.8]

# Cluster
ABM = [0.435136, 0.92979, 1.83469, 21.4203, 86.8384, 626.868, 6656.62, 66702.9, 110549, 292610]
PDMM = [0.0209028, 0.134041, 0.261904, 1.5758, 3.01874, 6.36454, 13.0204, 27.0013, 30.5287, 41.0374]
Hybrid = [0.530828, 1.13194, 2.00169, 13.7976, 44.2729, 247.372, 1309.15, 10986.2, 23386.3, 59637]

fig, ax = plt.subplots()
ax.plot(x, ABM, marker='s', label='ABM', color='indianred')
ax.plot(x, Hybrid, marker='o', label='Spatial Hybrid', color='royalblue')
ax.plot(x, PDMM, marker='x', label='PDMM', color='green')
ax.set_xlabel('Number of Agents')
ax.set_ylabel('Time(seconds)')
ax.legend()
plt.yscale("log")
plt.xscale("log")
ax.grid(True, linestyle=':')
fig.subplots_adjust(bottom = 0.15)
fig.subplots_adjust(left=0.15)
fig.savefig("time.png")