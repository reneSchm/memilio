from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
from matplotlib.colors import Normalize


# def f(x, y):
#     return (x**2-1)**2 + (y**2-1)**2

def f(x, y):
    return (x**4+y**4)/2

font_size = 16

plt.rc ('font', size = font_size) # steuert die Standardtextgröße
plt.rc ('axes', titlesize = font_size) # Schriftgröße des Titels
plt.rc ('axes', labelsize = font_size) # Schriftgröße der x- und y-Beschriftungen
plt.rc ('xtick', labelsize = font_size) #Schriftgröße der x-Tick-Labels
plt.rc ('ytick', labelsize = font_size) #Schriftgröße der y-Tick-Labels
plt.rc ('legend', fontsize = font_size) #Schriftgröße der Legende


x = np.linspace(-2, 2, 100)
y = np.linspace(-2, 2, 100)
X, Y = np.meshgrid(x, y)

colors = 'RdYlGn'

fig, ax = plt.subplots()

hl = ax.contour(X, Y, f(X, Y), [0, 0.25, 0.5,
                0.75, 1, 1.25, 1.5, 1.75, 2], cmap=colors, vmin=0, vmax=2)
plt.clabel(hl, inline=1, fontsize=10)
norm = Normalize(vmin=0, vmax=2)
cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=colors), ax=ax)
ax.set_xlim([-1.75, 1.75])
ax.set_ylim([-1.75, 1.75])
fig.savefig('plt.png')


# x = np.linspace(-1.5, 1.5, 200)
# y = np.linspace(-1.5, 1.5, 200)
x = np.linspace(-3, 3, 400)
y = np.linspace(-3, 3, 400)
X, Y = np.meshgrid(x, y)

fig1 = plt.figure()
ax1 = fig1.add_subplot(111, projection='3d', proj_type = 'ortho')
p = ax1.plot_surface(X, Y, f(X, Y), cmap=colors)
fig1.colorbar(p, shrink=1, aspect=20)
# ax1.set_xlim([-1.5, 1.5])
# ax1.set_ylim([-1.5, 1.5])
ax1.set_xlim([-3, 3])
ax1.set_ylim([-3, 3])
#ax1.set_zlim([0,2])

ax1.elev = 45
ax1.azim = -135

fig1.savefig('plt1.png', bbox_inches='tight')
