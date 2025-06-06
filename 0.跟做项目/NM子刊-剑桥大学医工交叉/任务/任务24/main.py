import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import mean_absolute_error
from matplotlib import cm
from scipy.ndimage import gaussian_filter

def patchwise_alignment_smooth_eval(
    he_img, st_pred, st_gt,
    patch_size=32, stride=None, ssim_threshold=0.7,
    save_prefix='./result', apply_gradient=False
):
    """
    滑窗SSIM评估 + Checkerboard渐变图 + 小提琴图评估鲁棒性

    参数:
        he_img: ndarray, HE图像 (灰度或RGB)
        st_pred: ndarray, 模型预测ST图像 (H, W)
        st_gt: ndarray, Ground truth ST图像 (H, W)
        patch_size: int, 每个patch的尺寸 (默认32)
        stride: int, 滑动窗口步长 (默认patch_size // 2)
        ssim_threshold: float, 判断区域是否对齐良好的阈值
        save_prefix: str, 保存文件前缀
        apply_gradient: bool, 是否使用HE图的梯度图作为背景
    """
    H, W = st_gt.shape
    if stride is None:
        stride = patch_size // 2

    # HE背景图处理
    if apply_gradient:
        he_gray = cv2.cvtColor(he_img, cv2.COLOR_RGB2GRAY) if he_img.ndim == 3 else he_img
        grad_x = cv2.Sobel(he_gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(he_gray, cv2.CV_64F, 0, 1, ksize=3)
        he_base = np.uint8(np.clip(np.sqrt(grad_x**2 + grad_y**2), 0, 255))
        he_base = cv2.cvtColor(he_base, cv2.COLOR_GRAY2RGB)
    else:
        he_base = cv2.cvtColor(he_img, cv2.COLOR_GRAY2RGB) if he_img.ndim == 2 else he_img.copy()

    # 初始化SSIM与性能图
    ssim_map = np.zeros((H, W))
    weight_map = np.zeros((H, W))
    error_list = []
    region_type = []

    for i in range(0, H - patch_size + 1, stride):
        for j in range(0, W - patch_size + 1, stride):
            pred_patch = st_pred[i:i+patch_size, j:j+patch_size]
            gt_patch = st_gt[i:i+patch_size, j:j+patch_size]

            ssim_val = ssim(pred_patch, gt_patch, data_range=gt_patch.max() - gt_patch.min())
            mae_val = mean_absolute_error(gt_patch.flatten(), pred_patch.flatten())

            # 添加高斯平滑（区域中心权重大）
            weight = np.outer(
                cv2.getGaussianKernel(patch_size, patch_size/6),
                cv2.getGaussianKernel(patch_size, patch_size/6)
            )

            ssim_map[i:i+patch_size, j:j+patch_size] += ssim_val * weight
            weight_map[i:i+patch_size, j:j+patch_size] += weight

            region_type.append("aligned" if ssim_val >= ssim_threshold else "misaligned")
            error_list.append(mae_val)

    # 归一化得到平滑SSIM图
    smooth_ssim = ssim_map / (weight_map + 1e-8)
    smooth_ssim = np.clip(smooth_ssim, 0, 1)

    # 可视化：叠加渐变颜色图
    cmap = cm.get_cmap("RdYlGn")  # 红-黄-绿
    ssim_color = cmap(smooth_ssim)[:, :, :3]  # RGB
    ssim_overlay = (ssim_color * 255).astype(np.uint8)

    alpha = 0.6
    blended = cv2.addWeighted(ssim_overlay, alpha, he_base, 1 - alpha, 0)

    plt.figure(figsize=(8, 8))
    plt.imshow(blended)
    plt.title("Smooth Checkerboard SSIM Overlay")
    plt.axis("off")
    plt.savefig(f"{save_prefix}_checkerboard.png", bbox_inches="tight")
    plt.close()

    # 小提琴图：aligned vs misaligned 区域性能差异
    plt.figure(figsize=(6, 5))
    sns.violinplot(x=region_type, y=error_list, palette={"aligned": "green", "misaligned": "red"})
    plt.title("Model Performance in Aligned vs Misaligned Regions")
    plt.ylabel("MAE")
    plt.xlabel("Region Type")
    plt.savefig(f"{save_prefix}_violin.png", bbox_inches="tight")
    plt.close()
