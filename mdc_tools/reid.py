import albumentations as albu
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
import cv2

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

