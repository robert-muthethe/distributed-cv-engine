import cv2
import numpy as np

print("Initializing Day 2: PnP Pose Estimation")

# 1. Define the 3D geometry of the object (e.g., a tracking target with 4 markers)
# Coordinates are in millimeters in the object's local frame
object_points = np.array([
    [0.0, 0.0, 0.0],      # Marker 1 (Origin)
    [100.0, 0.0, 0.0],    # Marker 2
    [100.0, 100.0, 0.0],  # Marker 3
    [0.0, 100.0, 0.0]     # Marker 4
], dtype=np.float32)

# 2. Define the Camera Intrinsic Matrix (K)
# Simulating a standard camera with focal length = 800 pixels and center at (640, 480)
focal_length = 800.0
center_x, center_y = 640.0, 480.0
camera_matrix = np.array([
    [focal_length, 0.0, center_x],
    [0.0, focal_length, center_y],
    [0.0, 0.0, 1.0]
], dtype=np.float32)

# Assuming zero lens distortion for the mathematical model
dist_coeffs = np.zeros((4, 1), dtype=np.float32)

# 3. Define Ground Truth Pose (True camera position and orientation)
# True rotation vector (axis-angle representation)
true_rvec = np.array([[0.1], [0.25], [-0.15]], dtype=np.float32)
# True translation vector (Camera is 500mm away along Z axis, slightly offset)
true_tvec = np.array([[10.0], [-20.0], [500.0]], dtype=np.float32)

# 4. Generate simulated 2D pixel points by projecting the 3D points
print("Projecting 3D object points to 2D image plane...")
image_points, _ = cv2.projectPoints(object_points, true_rvec, true_tvec, camera_matrix, dist_coeffs)

# Add minor pixel noise to simulate a real sensor
np.random.seed(42)
noise = np.random.normal(0, 0.5, image_points.shape).astype(np.float32)
image_points_noisy = image_points + noise

# 5. Solve the PnP problem to estimate the pose from 2D-3D correspondences
print("Solving PnP configuration using OpenCV...")
success, estimated_rvec, estimated_tvec = cv2.solvePnP(
    object_points, 
    image_points_noisy, 
    camera_matrix, 
    dist_coeffs, 
    flags=cv2.SOLVEPNP_ITERATIVE
)

# 6. Convert the rotation vector to a 3x3 rotation matrix using the Rodrigues formula
estimated_R, _ = cv2.Rodrigues(estimated_rvec)

# Print results and errors
print("\n--- Evaluation Results ---")
print(f"PnP Solver Success Status: {success}")
print("\nTrue Translation Vector (Ground Truth):")
print(true_tvec.flatten())
print("Estimated Translation Vector:")
print(estimated_tvec.flatten())

translation_error = np.linalg.norm(true_tvec - estimated_tvec)
print(f"\nAbsolute Translation Error: {translation_error:.4f} mm")