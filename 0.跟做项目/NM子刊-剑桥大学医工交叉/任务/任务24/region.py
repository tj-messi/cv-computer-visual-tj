import numpy as np
import cv2
import matplotlib.pyplot as plt

def generate_simulated_data_local():
    # 高分辨H&E模拟图 (256x256, RGB)
    he_img = np.zeros((256, 256, 3), dtype=np.uint8)
    cv2.rectangle(he_img, (30, 30), (100, 100), (180, 120, 200), -1)
    cv2.rectangle(he_img, (150, 150), (220, 220), (120, 180, 130), -1)

    # 细胞分割二值图 (0/1)
    seg_mask = np.zeros((256, 256), dtype=np.uint8)
    cv2.rectangle(seg_mask, (35, 35), (95, 95), 1, -1)
    cv2.rectangle(seg_mask, (155, 155), (215, 215), 1, -1)

    # ST点（相对于局部图）
    st_points = np.array([
        [40, 40],
        [90, 90],
        [160, 160],
        [210, 210]
    ])

    # 细胞中心点，轻微偏移模拟
    cell_centers = st_points + np.random.uniform(-3, 3, st_points.shape)

    return he_img, seg_mask, st_points, cell_centers

def compute_gradient_binary(mask):
    blurred = cv2.GaussianBlur(mask.astype(np.float32), (5, 5), 0)
    grad_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    return grad_x, grad_y

def sample_gradient_at_points(grad_x, grad_y, points):
    sampled_vectors = []
    h, w = grad_x.shape
    for (x, y) in points:
        x0, y0 = int(np.floor(x)), int(np.floor(y))
        x1, y1 = min(x0 + 1, w - 1), min(y0 + 1, h - 1)

        dx = x - x0
        dy = y - y0

        gx = (1 - dx) * (1 - dy) * grad_x[y0, x0] + dx * (1 - dy) * grad_x[y0, x1] + (1 - dx) * dy * grad_x[y1, x0] + dx * dy * grad_x[y1, x1]
        gy = (1 - dx) * (1 - dy) * grad_y[y0, x0] + dx * (1 - dy) * grad_y[y0, x1] + (1 - dx) * dy * grad_y[y1, x0] + dx * dy * grad_y[y1, x1]

        sampled_vectors.append((gx, gy))
    return np.array(sampled_vectors)

def visualize_vector_field_local(he_img, st_points, cell_centers, vectors, title, svg_save_path):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(he_img)

    offsets = st_points - cell_centers
    combined_vectors = offsets + vectors * 5  # 放大梯度向量

    for (x, y), (dx, dy) in zip(cell_centers, combined_vectors):
        ax.arrow(x, y, dx, dy,
                 head_width=3, head_length=4, fc='blue', ec='blue', alpha=0.9, length_includes_head=True)

    ax.scatter(st_points[:, 0], st_points[:, 1], color='yellow', label='ST points', s=25)
    ax.scatter(cell_centers[:, 0], cell_centers[:, 1], color='red', label='Cell centers', s=25)

    ax.set_title(title)
    ax.axis('off')
    ax.legend()

    plt.tight_layout()
    plt.savefig(svg_save_path)
    plt.show()

def main_local():
    he_img, seg_mask, st_points, cell_centers = generate_simulated_data_local()

    grad_x_seg, grad_y_seg = compute_gradient_binary(seg_mask)
    grad_x_st, grad_y_st = grad_x_seg, grad_y_seg  # 简化，真实应用请替换

    vectors_seg = sample_gradient_at_points(grad_x_seg, grad_y_seg, st_points)
    vectors_st = sample_gradient_at_points(grad_x_st, grad_y_st, st_points)

    deformation_vectors = vectors_st - vectors_seg

    visualize_vector_field_local(he_img, st_points, cell_centers, deformation_vectors,
                                'Local Scale: Deformation Vector Field Overlay',
                                'local_deformation_vector_field.svg')

if __name__ == "__main__":
    main_local()
