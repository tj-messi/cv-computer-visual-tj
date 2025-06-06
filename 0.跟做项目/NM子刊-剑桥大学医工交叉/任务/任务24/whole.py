import numpy as np
import cv2
import matplotlib.pyplot as plt

def generate_simulated_data_wsi():
    # 生成模拟H&E图 (512x512, RGB)
    he_img = np.zeros((512, 512, 3), dtype=np.uint8)
    cv2.circle(he_img, (150, 200), 100, (200, 150, 100), -1)
    cv2.circle(he_img, (350, 300), 120, (150, 200, 120), -1)

    # 生成模拟ST点 (N, 2)，在图内随机分布
    st_points = np.array([
        [140, 190],
        [160, 210],
        [355, 310],
        [340, 280],
        [300, 350]
    ])

    # 细胞中心，假设每个ST点对应一个细胞中心 (轻微偏移模拟)
    cell_centers = st_points + np.random.uniform(-5, 5, st_points.shape)

    return he_img, st_points, cell_centers

def compute_gradient(image_gray):
    # 使用cv2.Sobel计算梯度
    grad_x = cv2.Sobel(image_gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(image_gray, cv2.CV_64F, 0, 1, ksize=3)
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

def visualize_vector_field(he_img, st_points, cell_centers, vectors, title, svg_save_path):
    import matplotlib.patches as patches

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(he_img)

    offsets = st_points - cell_centers
    combined_vectors = offsets + vectors * 2  # 放大梯度向量以便可视化

    for (x, y), (dx, dy) in zip(cell_centers, combined_vectors):
        ax.arrow(x, y, dx, dy,
                 head_width=5, head_length=7, fc='red', ec='red', alpha=0.8, length_includes_head=True)

    ax.scatter(st_points[:, 0], st_points[:, 1], color='yellow', label='ST points', s=30)
    ax.scatter(cell_centers[:, 0], cell_centers[:, 1], color='cyan', label='Cell centers', s=30)

    ax.set_title(title)
    ax.axis('off')
    ax.legend()

    plt.tight_layout()
    plt.savefig(svg_save_path)
    plt.show()

def main_wsi():
    he_img, st_points, cell_centers = generate_simulated_data_wsi()
    he_gray = cv2.cvtColor(he_img, cv2.COLOR_RGB2GRAY)

    grad_x_he, grad_y_he = compute_gradient(he_gray)
    grad_x_st, grad_y_st = compute_gradient(he_gray)  # 模拟：这里用同图替代ST图像梯度

    vectors_he = sample_gradient_at_points(grad_x_he, grad_y_he, st_points)
    vectors_st = sample_gradient_at_points(grad_x_st, grad_y_st, st_points)

    deformation_vectors = vectors_st - vectors_he

    visualize_vector_field(he_img, st_points, cell_centers, deformation_vectors,
                           'WSI Scale: Deformation Vector Field Overlay',
                           'wsi_deformation_vector_field.svg')

if __name__ == "__main__":
    main_wsi()
