import numpy as np

def douglas_peucker(points, epsilon):
    """
    Simplify a path represented by an array of points using the Douglas-Peucker algorithm.

    Args:
        points (ndarray): Array of shape (n, d) containing n points in d dimensions.
        epsilon (float): The maximum distance between the simplified path and the original path.

    Returns:
        ndarray: Array of shape (m, d) containing m points in d dimensions representing the simplified path.
    """
    # Ensure input is a NumPy array
    points = np.asarray(points)

    # Check if we can simplify further
    if len(points) < 3:
        return points

    # Calculate distance between each point and the line formed by the first and last points
    dmax = 0
    index = 0
    for i in range(1, len(points) - 1):
        d = np.linalg.norm(np.cross(points[i] - points[0], points[-1] - points[0])) / np.linalg.norm(points[-1] - points[0])
        if d > dmax:
            index = i
            dmax = d

    # If the maximum distance is greater than epsilon, recursively simplify both subpaths
    if dmax > epsilon:
        first_simplified = douglas_peucker(points[:index+1], epsilon)
        second_simplified = douglas_peucker(points[index:], epsilon)[1:]
        return np.vstack((first_simplified[:-1], second_simplified))

    # Otherwise, return the endpoints of the path
    else:
        return np.vstack((points[0], points[-1]))


import numpy as np
import matplotlib.pyplot as plt

# from douglas_peucker import douglas_peucker

# Generate a random path in 3D space
np.random.seed(12345)
path = np.random.rand(50, 3)

# Simplify the path using the Douglas-Peucker algorithm with epsilon=0.1
simplified_path = douglas_peucker(path, 0.1)

# Plot the original and simplified paths
fig, ax = plt.subplots()
ax.scatter(path[:, 0], path[:, 1], path[:, 2], label='Original')
ax.plot(simplified_path[:, 0], simplified_path[:, 1], simplified_path[:, 2], c='r', label='Simplified')
ax.set_xlabel('X')
ax.set_ylabel('Y')
# ax.set_zlabel('Z')
ax.legend()
plt.show()
