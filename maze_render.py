import numpy as np
import matplotlib.pyplot as plt

# array index is vertical, then horizontal
enemy_list = (2, 3, 6, 15, 17, 19, 24, 26, 28, 29, 31, 33, 35, 37, 40, 42, 46, 49, 55, 57, 58, 61, 77, 78)
maze = np.ones((9, 9))
for each in enemy_list:
    maze[each // 9, each % 9] = 0

maze[8, 8] = 0.5
maze[0, 0] = 0.75
plt.imshow(maze, cmap="gray")
plt.show()


