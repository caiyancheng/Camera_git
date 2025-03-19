import numpy as np
import cv2
from scipy.ndimage import gaussian_filter
from scipy.optimize import minimize
from skimage import color

# Soft-threshold shrinkage operator
def soft_threshold(x, threshold):
    return np.sign(x) * np.maximum(np.abs(x) - threshold, 0)

# Polyphase decomposition into 4 components
def polyphase_decompose(img):
    return [img[0::2, 0::2], img[0::2, 1::2], img[1::2, 0::2], img[1::2, 1::2]]

# Polyphase reconstruction from 4 components
def polyphase_reconstruct(components):
    shape = (components[0].shape[0] * 2, components[0].shape[1] * 2)
    out = np.zeros(shape, dtype=components[0].dtype)
    out[0::2, 0::2], out[0::2, 1::2], out[1::2, 0::2], out[1::2, 1::2] = components
    return out

# Patch extraction (for simplicity, non-overlapping 8x8)
def extract_patches(img, patch_size=8):
    patches = []
    for y in range(0, img.shape[0] - patch_size + 1, patch_size):
        for x in range(0, img.shape[1] - patch_size + 1, patch_size):
            patches.append(img[y:y+patch_size, x:x+patch_size].flatten())
    return np.array(patches)

# GMM simulation (use pretrained GMM weights or dummy normal priors)
def gmm_log_likelihood(patch, mean=0, cov=1):
    return -0.5 * np.sum((patch - mean)**2) / (cov ** 2)

# Local Laplacian Filter approximation (using bilateral filter fallback)
def local_laplacian_filter(img):
    return cv2.bilateralFilter(img.astype(np.float32), d=9, sigmaColor=0.1, sigmaSpace=15)

# LDPC single decomposition step (simplified)
def ldpc_layer_decomposition(img, iterations=5, alpha=0.025, sigma=0.005):
    B = img.copy()
    M = np.zeros_like(img)
    for _ in range(iterations):
        residual = img - B - M
        grad_B = np.gradient(B)
        # Shrink gradient (D-subproblem)
        D_x = soft_threshold(grad_B[0], alpha)
        D_y = soft_threshold(grad_B[1], alpha)

        # Update B and M with a simple proximal step
        B = B + 0.1 * (residual + (D_x + D_y))
        M = img - B  # Simple update rule

        # Clip values
        B = np.clip(B, 0, 1)
        M = np.clip(M, 0, 1)
    return B, M

# Main LDPC pipeline
def ldpc_denoise_pipeline(img_rgb):
    img_yuv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YUV).astype(np.float32) / 255.0
    Y, U, V = img_yuv[...,0], img_yuv[...,1], img_yuv[...,2]

    # Decompose Y into polyphase components
    components = polyphase_decompose(Y)
    bg_components = []

    for comp in components:
        B, M = ldpc_layer_decomposition(comp)
        bg_components.append(B)
    Y_bg = polyphase_reconstruct(bg_components)

    # Local Laplacian filter
    Y_smooth = local_laplacian_filter(Y_bg)

    # Repeat on each RGB channel
    img_filtered = cv2.cvtColor(np.stack([Y_smooth, U, V], axis=-1), cv2.COLOR_YUV2RGB)

    R, G, B = img_filtered[...,0], img_filtered[...,1], img_filtered[...,2]
    final_R, _ = ldpc_layer_decomposition(R)
    final_G, _ = ldpc_layer_decomposition(G)
    final_B, _ = ldpc_layer_decomposition(B)

    final_img = np.stack([final_R, final_G, final_B], axis=-1)
    final_img = (np.clip(final_img, 0, 1) * 255).astype(np.uint8)
    return final_img

# Example usage
if __name__ == "__main__":
    input_img = cv2.imread("demoire_input.png")
    input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)

    output_img = ldpc_denoise_pipeline(input_img)

    cv2.imwrite("demoire_output.png", cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR))