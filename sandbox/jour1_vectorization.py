import numpy as np
import time

# Scale up to 1,000,000 points to ensure measurable execution time
print("Generating test dataset...")
n_points = 1000000
points_3d = np.random.rand(n_points, 3) * 100

# Define a 4x4 homogeneous transformation matrix
# 45-degree rotation around Y-axis + translation [10, -5, 2]
theta = np.radians(45)
c, s = np.cos(theta), np.sin(theta)
T = np.array([
    [c,  0, s, 10.0],
    [0,  1, 0, -5.0],
    [-s, 0, c,  2.0],
    [0,  0, 0,  1.0]
])

# Method A: Iterative Loop (Slow)
print("Starting iterative calculation (for loop)...")
# Using high-resolution performance counter
t0 = time.perf_counter()

# We only loop over the first 50,000 points to avoid waiting too long
# A full 1,000,000 loop in Python would take too much time
n_loop = 50000
points_transformed_loop = np.zeros((n_loop, 3))
for i in range(n_loop):
    p_homogeneous = np.array([points_3d[i, 0], points_3d[i, 1], points_3d[i, 2], 1.0])
    p_transformed = np.dot(T, p_homogeneous)
    points_transformed_loop[i] = p_transformed[:3]

t_loop = time.perf_counter() - t0
# Estimate total loop time for 1,000,000 points for fair comparison
estimated_total_loop_time = t_loop * (n_points / n_loop)
print(f"Iterative loop execution time (scaled to 1M points): {estimated_total_loop_time:.5f} seconds")

# Method B: Pure Vectorization with NumPy (Fast)
print("Starting vectorized calculation (NumPy)...")
t0 = time.perf_counter()

# Add a column of ones to convert all points to homogeneous coordinates
ones = np.ones((n_points, 1))
points_homogeneous = np.hstack((points_3d, ones))

# Perform global matrix multiplication: (n_points, 4) x (4, 4)^T
points_transformed_numpy = (points_homogeneous @ T.T)[:, :3]

t_numpy = time.perf_counter() - t0
print(f"Vectorized NumPy execution time: {t_numpy:.5f} seconds")

# Verify mathematical consistency on the looped subset
assert np.allclose(points_transformed_loop, points_transformed_numpy[:n_loop]), "Error: Results do not match"
print("Validation: Both methods returned identical results.")

# Prevent division by zero if NumPy time is still extremely low
if t_numpy > 0:
    print(f"Performance Gain: NumPy is {estimated_total_loop_time / t_numpy:.1f}x faster than the loop.")
else:
    print("NumPy execution time was too fast to measure accurately with the CPU clock.")