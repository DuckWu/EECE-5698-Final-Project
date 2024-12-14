import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

def load_artRoom_calibration():
    cameras = {
        0: {
            'K': np.array([[1733.74, 0, 792.27],
                           [0, 1733.74, 541.89],
                           [0, 0, 1]]),
            'R': np.eye(3),
            't': np.zeros(3)
        },
        1: {
            'K': np.array([[1733.74, 0, 792.27],
                           [0, 1733.74, 541.89],
                           [0, 0, 1]]),
            'R': np.eye(3),
            't': np.array([-536.62, 0, 0])
        }
    }
    return cameras, 0, 1, 1920, 1080, 170, 55, 142

def compute_psnr(gt, approx):
    h = min(gt.shape[0], approx.shape[0])
    w = min(gt.shape[1], approx.shape[1])
    gt = gt[:h, :w]
    approx = approx[:h, :w]

    valid_mask = np.isfinite(gt) & np.isfinite(approx)
    if not np.any(valid_mask):
        return float('inf')
    
    mse = np.mean((gt[valid_mask] - approx[valid_mask]) ** 2)
    if mse == 0:
        return float('inf')
    
    max_pixel = 255.0
    psnr = 10 * np.log10((max_pixel ** 2) / mse)
    return psnr

def read_pfm(file):
    with open(file, 'rb') as f:
        header = f.readline().decode('latin-1').strip()
        if header == 'PF':
            color = True
        elif header == 'Pf':
            color = False
        else:
            raise Exception("Not a PFM file.")

        dims = f.readline().decode('latin-1').strip().split()
        width, height = int(dims[0]), int(dims[1])
        scale_line = f.readline().decode('latin-1').strip()
        scale = float(scale_line)
        if scale < 0: # little-endian
            endian = '<'
            scale = -scale
        else:
            endian = '>'

        data = np.frombuffer(f.read(), endian + 'f')
        if color:
            data = np.reshape(data, (height, width, 3))
        else:
            data = np.reshape(data, (height, width))
        data = np.flipud(data)
        return data, scale

def quantize_block(block, qstep):
    block_q = np.round(block / qstep) * qstep
    return block_q

def compute_entropy(block):
    hist, _ = np.histogram(block, bins=256, range=(0,255))
    p = hist.astype(np.float32)
    s = np.sum(p)
    if s == 0:
        return 0.0
    p = p / s
    p = p[p > 0]
    entropy = -np.sum(p * np.log2(p))
    return entropy

def rate_distortion_optimization(disparity_map, block_size, lambda_value, quant_steps=[2, 4, 6, 10, 12, 24]):
    h, w = disparity_map.shape
    optimized_map = np.zeros_like(disparity_map)

    for y in range(0, h, block_size):
        for x in range(0, w, block_size):
            block = disparity_map[y:y+block_size, x:x+block_size]
            if block.size == 0:
                continue

            best_cost = float('inf')
            best_block_approx = None

            for qstep in quant_steps:
                block_q = quantize_block(block, qstep)
                mse = np.mean((block - block_q)**2)
                block_q_clipped = np.clip(block_q, 0, 255).astype(np.uint8)
                rate = compute_entropy(block_q_clipped)
                cost = mse + lambda_value * rate

                if cost < best_cost:
                    best_cost = cost
                    best_block_approx = block_q

            optimized_map[y:y+block_size, x:x+block_size] = best_block_approx

    return optimized_map

# Create directories for saving images if they don't exist
os.makedirs('images', exist_ok=True)
os.makedirs('plots', exist_ok=True)

# Load calibration data
cameras, ref_cam_id, sec_cam_id, width, height, ndisp, vmin, vmax = load_artRoom_calibration()

ref_cam = cameras[ref_cam_id]
sec_cam = cameras[sec_cam_id]

K1 = ref_cam['K']
R1 = ref_cam['R']
t1 = ref_cam['t']

K2 = sec_cam['K']
R2 = sec_cam['R']
t2 = sec_cam['t']

# Paths (adjust as needed)
ref_img_path = r'E:\all\data\artroom1\im0.png'
sec_img_path = r'E:\all\data\artroom1\im1.png'
gt_disp_path = r'E:\all\data\artroom1\disp0.pfm'  # Ground truth disparity

# Load images
ref_img = cv2.imread(ref_img_path, cv2.IMREAD_COLOR)
sec_img = cv2.imread(sec_img_path, cv2.IMREAD_COLOR)

if ref_img is None or sec_img is None:
    raise IOError("Cannot load images. Check paths and filenames.")

# Read ground truth disparity
gt_disparity, gt_scale = read_pfm(gt_disp_path)

# Compute relative rotation and translation
R_rel = R2 @ R1.T
t_rel = t2 - R_rel @ t1

# Stereo rectification
image_size = (width, height)
R1_rect, R2_rect, P1_rect, P2_rect, Q, _, _ = cv2.stereoRectify(
    cameraMatrix1=K1,
    distCoeffs1=None,
    cameraMatrix2=K2,
    distCoeffs2=None,
    imageSize=image_size,
    R=R_rel,
    T=t_rel
)
map1x, map1y = cv2.initUndistortRectifyMap(K1, None, R1_rect, P1_rect, image_size, cv2.CV_32FC1)
map2x, map2y = cv2.initUndistortRectifyMap(K2, None, R2_rect, P2_rect, image_size, cv2.CV_32FC1)

ref_img_rect = cv2.remap(ref_img, map1x, map1y, cv2.INTER_LINEAR)
sec_img_rect = cv2.remap(sec_img, map2x, map2y, cv2.INTER_LINEAR)

gray_ref = cv2.cvtColor(ref_img_rect, cv2.COLOR_BGR2GRAY)
gray_sec = cv2.cvtColor(sec_img_rect, cv2.COLOR_BGR2GRAY)

# Stereo matching
window_size = 10
min_disp = 0
num_disp = ndisp

stereo = cv2.StereoSGBM_create(
    minDisparity=min_disp,
    numDisparities=num_disp,
    blockSize=15,
    P1=8 * 3 * window_size ** 2,
    P2=32 * 3 * window_size ** 2,
    uniquenessRatio=10,
    speckleWindowSize=100,
    speckleRange=32,
    disp12MaxDiff=1
)

disparity = stereo.compute(gray_ref, gray_sec).astype(np.float32) / 16.0
disparity_display = cv2.normalize(disparity, None, vmin, vmax, cv2.NORM_MINMAX)
disparity_display = np.uint8(disparity_display)

lambda_value = 1
block_size = 16
optimized_disparity = rate_distortion_optimization(disparity, block_size, lambda_value)

# Compute PSNR vs ground truth
psnr_original = compute_psnr(gt_disparity, disparity)
psnr_optimized = compute_psnr(gt_disparity, optimized_disparity)

print(f"PSNR of original disparity map: {psnr_original:.2f} dB")
print(f"PSNR of optimized disparity map: {psnr_optimized:.2f} dB")

# --- Save Figures ---
# Display and Save Rectified Images
plt.figure(figsize=(12, 6))

# Rectified Reference Image
plt.subplot(1, 2, 1)
plt.title("Rectified Reference Image")
plt.imshow(cv2.cvtColor(ref_img_rect, cv2.COLOR_BGR2RGB))
plt.axis('off')

# Rectified Secondary Image
plt.subplot(1, 2, 2)
plt.title("Rectified Secondary Image")
plt.imshow(cv2.cvtColor(sec_img_rect, cv2.COLOR_BGR2RGB))
plt.axis('off')

# Save the rectified images
cv2.imwrite("images/rectified_ref.png", ref_img_rect)
cv2.imwrite("images/rectified_sec.png", sec_img_rect)

plt.tight_layout()
plt.savefig("images/rectified_images.png")  # Save the combined figure
plt.show()
# 1. Ground Truth, Original, and Optimized Disparity
plt.figure(figsize=(18, 6))
plt.subplot(1, 3, 1)
plt.title("Ground Truth Disparity")
plt.imshow(gt_disparity, cmap='plasma')
plt.colorbar(label="Disparity Value")

plt.subplot(1, 3, 2)
plt.title(f"Original Disparity (PSNR: {psnr_original:.2f} dB)")
plt.imshow(disparity_display, cmap='plasma')
plt.colorbar(label="Disparity Value")

plt.subplot(1, 3, 3)
plt.title(f"Optimized Disparity (PSNR: {psnr_optimized:.2f} dB)")
plt.imshow(optimized_disparity, cmap='plasma')
plt.colorbar(label="Disparity Value")

plt.tight_layout()
plt.savefig("images/final_comparison.png")  # Save the figure
plt.show()

# 2. Extract and Save a Representative Block
# Choose a block from the center of the image (adjust x,y as needed)
bx, by = 600, 600  # block top-left corner
block_original = disparity[by:by+block_size, bx:bx+block_size]
block_optimized = optimized_disparity[by:by+block_size, bx:bx+block_size]

# Normalize blocks for display if needed
block_original_disp = cv2.normalize(block_original, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
block_optimized_disp = cv2.normalize(block_optimized, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

plt.figure(figsize=(8,4))
plt.subplot(1,2,1)
plt.title("Original Block")
plt.imshow(block_original_disp, cmap='plasma')
plt.colorbar()

plt.subplot(1,2,2)
plt.title("Optimized Block")
plt.imshow(block_optimized_disp, cmap='plasma')
plt.colorbar()

plt.tight_layout()
plt.savefig("images/block_comparison.png")
plt.show()

# 3. (Optional) If you vary lambda or quant steps in multiple runs,
# you can store PSNR and an entropy measure and plot a Rate-Distortion (R-D) curve.
# Here, we show an example of how you would plot if you had collected data.

# Example dummy data for demonstration:
lambdas = [0.1, 1, 5, 10]
psnr_values = [20.5, 21.2, 22.0, 22.3]
entropy_values = [6.5, 5.8, 5.2, 4.9]

plt.figure()
plt.plot(entropy_values, psnr_values, marker='o')
plt.title("Rate-Distortion Curve")
plt.xlabel("Entropy (bits/block)")
plt.ylabel("PSNR (dB)")
plt.grid(True)
plt.savefig("plots/rd_curve.png")
plt.show()

print("Images saved in 'images/' directory and plot in 'plots/' directory.")
