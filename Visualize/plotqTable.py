import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns

# Load the maze and the Q-table
maze = np.load('/home/tonix/Documents/Doctorado/Clases/Q-learning-Maze/Checkpoint/Maze.npy')
qtable = np.load('/home/tonix/Documents/Doctorado/Clases/Q-learning-Maze/Checkpoint/Q_table.npy')

# Reshape Q-table for plotting
size = int(np.sqrt(qtable.shape[0]))
reshaped_qtable = qtable.reshape((size, size, 4))

# Create a color map for the maze
cmap = mcolors.ListedColormap(['black', 'white'])  # Wall = black, Path = white
bounds = [0, 0.5, 1]
norm = mcolors.BoundaryNorm(bounds, cmap.N)

# Creating the figure and axes
fig = plt.figure(figsize=(15, 20))

# Add a subplot for the maze (larger)
ax_maze = fig.add_subplot(3, 2, (1, 2))  # Span the maze across the first two positions
ax_maze.imshow(maze, cmap=cmap, norm=norm)
ax_maze.set_title('Maze')
ax_maze.set_xticks([])
ax_maze.set_yticks([])

# Titles for Q-table levels
titles = ['up', 'down', 'left', 'right']

# Loop over each level and create a heatmap in a subplot
for i in range(4):
    ax = fig.add_subplot(3, 2, i + 3)  # Position the Q-table levels in the remaining slots
    sns.heatmap(reshaped_qtable[:, :, i], ax=ax, cmap='viridis')
    ax.set_title(titles[i])
    ax.set_xlabel('State Dimension 1')
    ax.set_ylabel('State Dimension 2')

# Adjust layout
plt.tight_layout()
plt.show()
