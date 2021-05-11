import torch
import cv2
import sys
def tensor2img(img, range=(0, 1), output='a.jpg'):
    """将任意tensor或ndarray存储为图片，方便调试使用

    Args:
        img (torch.Tensor): 任意形状或数据类型的tensor
    """
    if isinstance(img, torch.Tensor):
        img = torch.clamp((img + range[0]) / (range[1] - range[0]) * 255, 0, 255)
        if len(img.shape) == 4:
            img = torch.cat(*img, dim=0)
    
        if img.shape[0] == 3:
            img = img.permute([1, 2, 0])
        
        img = img.detach().cpu().numpy()
    
    cv2.imwrite(output, img)

if __name__ == "__main__":
    print(sys.argv)
    a = torch.randn(3, 4, 4)
    tensor2img(a) 