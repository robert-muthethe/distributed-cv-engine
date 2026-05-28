import numpy as np

print("Initializing Day 3: 3D Point Cloud Registration")

# 1. Generate a synthetic source point cloud (1000 points in 3D space)
np.random.seed(42)
n_points = 1000
source_cloud = np.random.rand(n_points, 3) * 50

# 2. Define a ground truth transformation (Rotation around X axis + Translation)
theta = np.radians(15)
c, s = np.cos(theta), np.sin(theta)
R_true = np.array([
    [1.0, 0.0, 0.0],
    [0.0, c, -s],
    [0.0, s, c]
])
t_true = np.array([2.5, -1.2, 5.0])

# 3. Create the target cloud by applying the transformation and adding noise
target_cloud = (source_cloud @ R_true.T) + t_true
sensor_noise = np.random.normal(0, 0.1, target_cloud.shape)
target_cloud += sensor_noise

# 4. Implement the Kabsch algorithm using SVD
# Step 4.1: Compute centroids of both point clouds
centroid_source = np.mean(source_cloud, axis=0)
centroid_target = np.mean(target_cloud, axis=0)

# Step 4.2: Center both point clouds
source_centered = source_cloud - centroid_source
target_centered = target_cloud - centroid_target

# Step 4.3: Compute the covariance matrix H
H = source_centered.T @ target_centered

# Step 4.4: Perform Singular Value Decomposition (SVD)
U, S, Vt = np.linalg.svd(H)

# Step 4.5: Calculate the optimal rotation matrix
R_estimated = Vt.T @ U.T

# Step 4.6: Handle special reflection case to ensure a valid rotation matrix
if np.linalg.det(R_estimated) < 0:
    Vt[2, :] *= -1
    R_estimated = Vt.T @ U.T

# Step 4.7: Calculate the optimal translation vector
t_estimated = centroid_target - (R_estimated @ centroid_source)

# 5. Evaluate the alignment performance
print("\n--- Evaluation Results ---")
print("True Translation Vector:")
print(t_true)
print("Estimated Translation Vector:")
print(t_estimated)

# Compute absolute translation error in space
registration_error = np.linalg.norm(t_true - t_estimated)
print(f"\nAbsolute Alignment Error: {registration_error:.4f} mm")