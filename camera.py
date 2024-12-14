import numpy as np
import matplotlib.pyplot as plt

def load_artRoom_calibration():
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


cameras, ref_cam_id, sec_cam_id, width, height, ndisp, vmin, vmax = load_artRoom_calibration()


camera_positions = [
    cameras[0]['t'],  
    cameras[1]['t'], 
]

camera_positions = np.array(camera_positions)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ref_pos = camera_positions[ref_cam_id] 
ax.scatter(ref_pos[0], ref_pos[1], ref_pos[2], c='g', marker='o', s=100, label=f'Ref Camera (Cam {ref_cam_id})')

for i, pos in enumerate(camera_positions):
    if i != ref_cam_id: 
        ax.scatter(pos[0], pos[1], pos[2], c='r', marker='o', label=f'Camera {i}')

for i, pos in enumerate(camera_positions):
    ax.text(pos[0], pos[1], pos[2], f"Cam {i}", color='blue')

ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
ax.set_zlabel("Z-axis")
ax.legend()
plt.title("Camera Positions")
plt.show()
