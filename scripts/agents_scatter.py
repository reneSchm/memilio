import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import gaussian_kde

agent_pre_positions = np.loadtxt("agents_pre.txt")
agent_post_positions = np.loadtxt("agents_post.txt")
# background = np.loadtxt("potentially_germany.pgm", skiprows=3)
fig = plt.figure(dpi=500)
marker_size=None#36./fig.dpi
plt.subplot(121)
# plt.imshow(background, cmap="gray")
plt.axis('off')
plt.tight_layout()
z = gaussian_kde(agent_pre_positions.T)(agent_pre_positions.T)
plt.scatter(agent_pre_positions[:, 0], agent_pre_positions[:, 1], c=z, s=marker_size, linewidths=0)
plt.subplot(122)
# plt.imshow(background, cmap="gray")
plt.axis('off')
plt.tight_layout()
z = gaussian_kde(agent_post_positions.T)(agent_post_positions.T)
plt.scatter(agent_post_positions[:, 0], agent_post_positions[:, 1], c=z, s=marker_size, linewidths=0)
# print(agent_positions.shape)
# for i in range(0, agent_positions.shape[1], 2):
#     plt.plot(agent_positions[:, i], background.shape[0] - agent_positions[:, i+1], linewidth=0.1)
plt.savefig("plt.png")