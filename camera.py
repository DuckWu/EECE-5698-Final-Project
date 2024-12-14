import numpy as np
import matplotlib.pyplot as plt

def load_artRoom_calibration():
    # This function provides intrinsic and extrinsic parameters for two cameras.
    cameras = {
        0: {
            'K': np.array([[1733.74, 0, 792.27],
                           [0, 1733.74, 541.89],
                           [0,    0,    1]]),
            'R': np.eye(3),
            't': np.zeros(3)
        },
        1: {
            'K': np.array([[1733.74, 0, 792.27],
                           [0, 1733.74, 541.89],
                           [0,    0,    1]]),
            'R': np.eye(3),
            't': np.array([-536.62, 0, 0])  # Baseline
        }
    }
    return cameras, 0, 1, 1920, 1080, 170, 55, 142

# Get calibration data
cameras, ref_cam_id, sec_cam_id, width, height, ndisp, vmin, vmax = load_artRoom_calibration()

# Extract camera positions based on extrinsic parameters
camera_positions = [
    cameras[0]['t'],  # Camera 0 position
    cameras[1]['t'],  # Camera 1 position
]

camera_positions = np.array(camera_positions)

# Plot the positions in 3D space
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Highlight the reference camera position differently
ref_pos = camera_positions[ref_cam_id]  # Reference camera position
ax.scatter(ref_pos[0], ref_pos[1], ref_pos[2], c='g', marker='o', s=100, label=f'Ref Camera (Cam {ref_cam_id})')

# Plot other cameras
for i, pos in enumerate(camera_positions):
    if i != ref_cam_id:  # Exclude reference camera
        ax.scatter(pos[0], pos[1], pos[2], c='r', marker='o', label=f'Camera {i}')

# Label each camera position
for i, pos in enumerate(camera_positions):
    ax.text(pos[0], pos[1], pos[2], f"Cam {i}", color='blue')

# Set axis labels and legend
ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
ax.set_zlabel("Z-axis")
ax.legend()
plt.title("Camera Positions")
plt.show()
