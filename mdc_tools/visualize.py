import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
import cv2
import albumentations as albu

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


def visualize_reid(query, galleries, query_pid, gallery_pids):
    transforms = albu.Compose(
        [
            # albu.SmallestMaxSize(256),
            albu.LongestMaxSize(256),
            # albu.CenterCrop(256, 256),
            albu.PadIfNeeded(min_height=256, min_width=256, border_mode=cv2.BORDER_CONSTANT, value=(150, 150, 150))
            # albu.PadIfNeeded(min_height=256, min_width=256, border_mode=cv2.BORDER_REPLICATE)
        ]
    )
    n = len(galleries)
    plt.figure(figsize=(4 * (n + 1), 5))
    plt.subplot(1, n + 1, 1)
    plt.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0, wspace=0, hspace=0)
    plt.xticks([])
    plt.yticks([])
    plt.title(query_pid)
    # query = cv2.resize(query, (256, 256))
    query = transforms(image=query)['image']
    plt.imshow(query)
    # plt.gca().add_patch(Rectangle((0, 0), query.shape[1], query.shape[0], edgecolor='w', linewidth=10, fill=False))
    for i in range(len(galleries)):
        g_img = galleries[i]
        # g_img = cv2.resize(g_img, (256,256))
        g_img = transforms(image=g_img)['image']
        g_pid = gallery_pids[i]
        plt.subplot(1, n + 1, i + 2)
        plt.xticks([])
        plt.yticks([])
        plt.title(g_pid)
        plt.imshow(g_img)
        if g_pid == query_pid:
            plt.gca().add_patch(
                Rectangle((0, 0), g_img.shape[1], g_img.shape[0], edgecolor='g', linewidth=10, fill=False))
        else:
            plt.gca().add_patch(
                Rectangle((0, 0), g_img.shape[1], g_img.shape[0], edgecolor='r', linewidth=10, fill=False))


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
