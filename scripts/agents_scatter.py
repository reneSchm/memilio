import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import gaussian_kde
import matplotlib.cm as cm
from matplotlib.colors import Normalize 

agent_pre_positions = np.loadtxt("Results/movement/positions800agents_50days_sigma0.3.txt")
#agent_post_positions = np.loadtxt("agents_post.txt")
# background = np.loadtxt("potentially_germany.pgm", skiprows=3)
fig = plt.figure(dpi=500)
marker_size=None#36./fig.dpi
plt.subplot() #121
# plt.imshow(background, cmap="gray")
fig , ax = plt.subplots()
#plt.axis('off')
plt.tight_layout()
z = gaussian_kde(agent_pre_positions.T)(agent_pre_positions.T)
colors = 'RdYlGn_r'

norm = Normalize(vmin = 0, vmax = 1)
plt.scatter(agent_pre_positions[:, 0], agent_pre_positions[:, 1], c=z, norm=norm, cmap=colors, s=marker_size, linewidths=0)
# cbar = fig.colorbar(cm.ScalarMappable(norm = norm, cmap=colors), ax=ax)
# cbar.ax.set_ylabel('Agent concentration')
#plt.clim(0,1)

# Set limits for the axes
ax.set_xlim([-2, 2])
ax.set_ylim([-2, 2])
#plt.subplot(122)
# plt.imshow(background, cmap="gray")
#plt.axis('off')
#plt.tight_layout()
#z = gaussian_kde(agent_post_positions.T)(agent_post_positions.T)
#plt.scatter(agent_post_positions[:, 0], agent_post_positions[:, 1], c=z, s=marker_size, linewidths=0)
# print(agent_positions.shape)
# for i in range(0, agent_positions.shape[1], 2):
#     plt.plot(agent_positions[:, i], background.shape[0] - agent_positions[:, i+1], linewidth=0.1)
plt.savefig("plt.png")