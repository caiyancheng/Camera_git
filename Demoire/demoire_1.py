## The method from paper: Demoireing for Screen-shot Images with  Multi-channel Layer Decomposition
import cv2
import numpy as np
from skimage import color
from scipy.ndimage import gaussian_filter
from sklearn.mixture import GaussianMixture
import joblib


# Polyphase decomposition (4 sub-images)
def polyphase_decompose(img):
    return [img[0::2, 0::2], img[0::2, 1::2], img[1::2, 0::2], img[1::2, 1::2]]


def polyphase_reconstruct(subs):
    h, w = subs[0].shape
    out = np.zeros((h * 2, w * 2), dtype=subs[0].dtype)
    out[0::2, 0::2] = subs[0]
    out[0::2, 1::2] = subs[1]
    out[1::2, 0::2] = subs[2]
    out[1::2, 1::2] = subs[3]
    return out


# Shrinkage operator for optimization
def shrinkage(x, thresh):
    return np.sign(x) * np.maximum(np.abs(x) - thresh, 0)


# Local Laplacian filter (approximation)
def local_laplacian_filter(img, sigma_m=0.09, sigma_d=0.4):
    return cv2.bilateralFilter((img * 255).astype(np.uint8), d=9, sigmaColor=25, sigmaSpace=25).astype(np.float32) / 255


# MAP estimation update based on GMM
# Approximate: choose best Gaussian component and Wiener filter
def map_update(patches, gmm_model, beta):
    n_patches = patches.shape[0]
    updated = np.zeros_like(patches)
    for i in range(n_patches):
        likelihoods = [
            gmm_model.weights_[k] *
            multivariate_normal.pdf(patches[i], mean=gmm_model.means_[k],
                                    cov=gmm_model.covariances_[k] + (1 / beta) * np.eye(patches.shape[1]))
            for k in range(gmm_model.n_components)
        ]
        k_star = np.argmax(likelihoods)
        Ck = gmm_model.covariances_[k_star]
        mk = gmm_model.means_[k_star]
        updated[i] = mk + (Ck @ np.linalg.inv(Ck + (1 / beta) * np.eye(Ck.shape[0]))) @ (patches[i] - mk)
    return updated


# Extract patches
def extract_patches(img, patch_size=8):
    h, w = img.shape
    patches = []
    for i in range(0, h - patch_size + 1, patch_size):
        for j in range(0, w - patch_size + 1, patch_size):
            patch = img[i:i + patch_size, j:j + patch_size].flatten()
            patches.append(patch)
    return np.array(patches)


# Restore patches to image
def restore_patches(patches, img_shape, patch_size=8):
    h, w = img_shape
    restored = np.zeros(img_shape)
    count = 0
    for i in range(0, h - patch_size + 1, patch_size):
        for j in range(0, w - patch_size + 1, patch_size):
            restored[i:i + patch_size, j:j + patch_size] = patches[count].reshape(patch_size, patch_size)
            count += 1
    return restored


# Layer decomposition
def layer_decomposition(img, gmm_b, gmm_m, alpha=0.025, beta_init=200, iter_num=5):
    B = np.copy(img)
    M = np.zeros_like(img)
    for _ in range(iter_num):
        D = shrinkage(np.gradient(B)[0], alpha / beta_init)
        fidelity = (img - B - M)
        B += fidelity * 0.5
        M += fidelity * 0.5

        # Update B with MAP from GMM background prior
        patches_B = extract_patches(B)
        updated_B = map_update(patches_B, gmm_b, beta_init)
        B = restore_patches(updated_B, img.shape)

        # Update M with MAP from GMM moiré prior
        patches_M = extract_patches(M)
        updated_M = map_update(patches_M, gmm_m, beta_init)
        M = restore_patches(updated_M, img.shape)

        beta_init *= 2
    return B, M


# Demoire pipeline
def demoire_image(img, gmm_b, gmm_m):
    img_yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV).astype(np.float32) / 255.0
    y, u, v = img_yuv[..., 0], img_yuv[..., 1], img_yuv[..., 2]

    subs_y = polyphase_decompose(y)
    b_subs_y = [layer_decomposition(sub, gmm_b, gmm_m)[0] for sub in subs_y]
    b_y = polyphase_reconstruct(b_subs_y)
    b_y_filtered = local_laplacian_filter(b_y)

    img_recon = cv2.cvtColor((np.stack([b_y_filtered, u, v], axis=-1) * 255).astype(np.uint8),
                             cv2.COLOR_YUV2RGB).astype(np.float32) / 255.0
    channels = cv2.split(img_recon)

    clean_channels = []
    for c in channels:
        subs_c = polyphase_decompose(c)
        b_subs_c = [layer_decomposition(sub, gmm_b, gmm_m)[0] for sub in subs_c]
        b_c = polyphase_reconstruct(b_subs_c)
        clean_channels.append(b_c)

    clean_img = cv2.merge(clean_channels)
    clean_img = np.clip(clean_img * 255, 0, 255).astype(np.uint8)
    return clean_img

# 示例：加载GMM模型
# gmm_background = joblib.load('gmm_background.pkl')
# gmm_moire = joblib.load('gmm_moire.pkl')

# 使用方法：
# image = cv2.imread('input_moire_image.png')
# image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# result = demoire_image(image_rgb, gmm_background, gmm_moire)
# cv2.imwrite('result.png', cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
