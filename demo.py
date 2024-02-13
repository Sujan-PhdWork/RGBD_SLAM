import numpy as np
from sklearn.neighbors import KDTree

# Generate two 300x3 matrices
matrix1 = np.random.rand(300, 3)
matrix2 = np.random.rand(300, 3)

# Sample 100 points from the first matrix
sample_indices = np.random.choice(matrix1.shape[0], 100, replace=False)
sample_points = matrix1[sample_indices]

# Build KDTree from the sampled points
kdtree = KDTree(matrix2)

# Find nearest neighbors in matrix2 for each sampled point
distances, indices = kdtree.query(sample_points, k=1)

# Print nearest neighbors and their distances
for i in range(len(sample_points)):
    print(f"Nearest neighbor of sample point {i} in matrix1 is point {indices[i]} in matrix2 with distance {distances[i]}")
