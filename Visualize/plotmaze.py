import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Load the solved maze from the .npy file
solved_maze = np.load('/home/tonix/Documents/Maze-solver-Q-learning/MazeGenerator/Dataset/maze.npy')

# Create a color map for the maze
cmap = plt.cm.plasma

# Normalize the values to use in the colormap
norm = mcolors.Normalize(vmin=np.min(solved_maze), vmax=np.max(solved_maze))

# Create the plot
fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(solved_maze, cmap=cmap, norm=norm)

# Customize the plot
ax.set_title('Solved Maze')
ax.set_xticks([])
ax.set_yticks([])  # Remove axis ticks

# Optional: Add grid lines to visualize the cells
ax.grid(which='both', linestyle='-', linewidth=1)
ax.set_xticks(np.arange(-0.5, len(solved_maze), 1), minor=True)
ax.set_yticks(np.arange(-0.5, len(solved_maze[0]), 1), minor=True)

#plt.savefig('mazesolved.png')
plt.show()