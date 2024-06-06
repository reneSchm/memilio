import numpy as np
import matplotlib.pyplot as plt

# Rows are timepoints and columns are positions per timepoint and agent
positions = np.loadtxt("pos_munich.txt")

fig, ax = plt.subplots()
background = np.loadtxt("../potentially_germany.pgm", skiprows=3)
background = np.where(background == 255, 0, background)
background = np.where(background > 0, 1, background)
plt.imshow(background, cmap='Greys')
# for agent in range(0, positions.shape[1], 2):
#     ax.plot(positions[:, agent], background.shape[0] - positions[:, agent+1], linewidth=0.2)


ax.plot(positions[:50, 50], background.shape[0] - positions[:50, 50+1], linewidth=0.4)
ax.plot(positions[:50, 4], background.shape[0] - positions[:50, 4+1], linewidth=0.4)
ax.plot(positions[:50, 12], background.shape[0] - positions[:50, 12+1], linewidth=0.4)
ax.plot(positions[:50, 14], background.shape[0] - positions[:50, 14+1], linewidth=0.4)
ax.plot(positions[:50, 20], background.shape[0] - positions[:50, 20+1], linewidth=0.4)
ax.plot(positions[:50, 30], background.shape[0] - positions[:50, 30+1], linewidth=0.4)
ax.plot(positions[:50, 42], background.shape[0] - positions[:50, 42+1], linewidth=0.4)
ax.plot(positions[:50, 52], background.shape[0] - positions[:50, 52+1], linewidth=0.4)
ax.plot(positions[:50, 70], background.shape[0] - positions[:50, 70+1], linewidth=0.4)
ax.plot(positions[:50, 86], background.shape[0] - positions[:50, 86+1], linewidth=0.4)
ax.plot(positions[:50, 96], background.shape[0] - positions[:50, 96+1], linewidth=0.4)
ax.plot(positions[:50, 104], background.shape[0] - positions[:50, 104+1], linewidth=0.4)
ax.plot(positions[:50, 122], background.shape[0] - positions[:50, 122+1], linewidth=0.4)
ax.plot(positions[:50, 136], background.shape[0] - positions[:50, 136+1], linewidth=0.4)

# # Move left y-axis and bottom x-axis to centre, passing through (0,0)
# ax.spines['left'].set_position('center')
# ax.spines['bottom'].set_position('center')

# # Eliminate upper and right axes
# ax.spines['right'].set_color('none')
# ax.spines['top'].set_color('none')

# # Show ticks in the left and lower axes only
# ax.xaxis.set_ticks_position('bottom')
# ax.yaxis.set_ticks_position('left')

# # ax.set_xlim([-2,2])
# # ax.set_ylim([-2,2])

# # Remove zero
# xticks = ax.xaxis.get_major_ticks()
# xticks[4].set_visible(False)
# yticks = ax.yaxis.get_major_ticks()
# yticks[4].set_visible(False)

fig.savefig('mvmnt.png', dpi=500)