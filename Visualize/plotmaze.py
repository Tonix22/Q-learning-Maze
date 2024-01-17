import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Load the maze
maze = np.load('Maze.npy')

# Create a color map for the maze
# Assuming 0 represents walls and 1 represents paths
cmap = mcolors.ListedColormap(['black', 'white'])  # Wall = black, Path = white
bounds = [0, 0.5, 1]
norm = mcolors.BoundaryNorm(bounds, cmap.N)

# Create the plot
fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(maze, cmap=cmap, norm=norm)

# Customize the plot
ax.set_title('Fancy Maze')
ax.set_xticks([])
ax.set_yticks([])  # Remove axis ticks

# Optional: Add grid lines to visualize the cells
ax.grid(which='both', color='gray', linestyle='-', linewidth=1)
ax.set_xticks(np.arange(-0.5, len(maze), 1), minor=True)
ax.set_yticks(np.arange(-0.5, len(maze[0]), 1), minor=True)

plt.show()
