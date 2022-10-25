import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import cv2

COLOR_LIST = [
    (0, 0, 0),
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (0, 255, 255),
    (255, 0, 255),
    (255, 255, 255),
]


# helper function for data visualization
def visualize_img(*no_title_images, rows=1, **images):
    """Plot images in one row."""
    n = len(images) + len(no_title_images)
    plt.figure(figsize=(16, 5 * rows))
    for i, image in enumerate(no_title_images):
        plt.subplot(rows, n // rows + 1, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(image)

    for i, (name, image) in enumerate(images.items()):
        plt.subplot(rows, n // rows + 1, len(no_title_images) + i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)

    plt.show()


def get_heatmap(weights, featuremap, image, alpha=0.3):
    """
    绘制heatmap
    :param np.ndarray weights: 不同层的权重 C
    :param np.ndarray featuremap: featuremap C,H,W
    :param np.ndarray image: 原图 H,W,3
    :return:
    """

    heatmap = np.sum(featuremap * weights.reshape([-1, 1, 1]), axis=0)  # [B, H, W]

    heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap)) * 255
    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    heatmap = heatmap.astype(np.uint8)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLORBRG2RGB)
    return (heatmap * alpha + image * ( 1 - alpha )).astype(np.uint8)


def render_mask_to_img(img, cls_map, num_classes):
    """

    :param img:
    :param cls_map:
    :return:
    """
    img = img.copy()
    for i in range(num_classes):
        if i == 0:
            continue
        img[cls_map == i] = img[cls_map == i] * 0.7 + np.array(COLOR_LIST[i]) * 0.3

    return img


def render_keypoints_to_img(image, points, kp_vis=None, diameter=5, color=(0,255,0)):
    if kp_vis is not None:
        points = [point for vis, point in zip(kp_vis, points) if vis]
    im = image.copy()

    for (x, y) in points:
        cv2.circle(im, (int(x), int(y)), diameter, color, -1)

    return im
