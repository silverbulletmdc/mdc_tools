import torch
import glob
import re
import os.path as osp
from PIL import Image
import cv2
import os
import numpy as np


def read_rgb_image(img_path, format='ndarray'):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            if format == 'PIL':
                img = Image.open(img_path).convert("RGB")
            elif format == 'ndarray':
                img = cv2.imread(img_path)
                if len(img.shape) == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
    return img


def load_checkpoint(output_dir, device="cpu", epoch=0, **kwargs):
    """
    含关键字model和optimizer会被正确加载到指定的设备上.
    如果不指定epoch，自动读取最大的epoch。

    :param output_dir:
    :param device:
    :param epoch:
    :param kwargs:
    :return:
    """

    # 不指定epoch则读取已保存的最大epoch
    if epoch == 0:
        for key in kwargs.keys():
            pths = glob.glob(f'{output_dir}/{key}_*.pth')
            epochs = [re.findall(rf'{output_dir}/{key}_([0-9]+)\.pth', name)[0] for name in pths]
            epochs = list(map(int, epochs))
            epoch = max(epochs)
            break

    for key, obj in kwargs.items():
        obj.load_state_dict(torch.load(f'{output_dir}/{key}_{epoch}.pth'))

        # move to target device
        if 'model' in key:
            obj.to(device)

        elif 'optimizer' in key:
            for state in obj.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device)

    return epoch + 1


def save_checkpoint(epoch, output_dir, **kwargs):
    for key, obj in kwargs.items():
        try:
            obj = obj.module
        except AttributeError:
            pass

        torch.save(obj.state_dict(), f'{output_dir}/{key}_{epoch}.pth')


def mkdir_p(path):
    if not os.path.exists(path):
        os.makedirs(path)
