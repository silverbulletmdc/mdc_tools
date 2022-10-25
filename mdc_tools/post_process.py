"""
图片到图片的对等映射数据集。

在进行各类inference操作时，往往会遵循固定的范式，即 
    读取数据->预处理->过网络->后处理目标图片->存储
其中，过网络这一步是没办法进行多进程加速的，只能通过增大batch_size来加速。

pytorch自带数据集可以在数据处理部分自动多线程。但是后处理往往需要用户自己实现。
这部分的工程量比较大，因此大多数情况下inference代码都是一张一张的跑，极大浪费了GPU资源。

本代码提供一种多进程实现以上pipeline的基类算法，使用者可以快速将之前的代码改为batch inference。
"""

from torch.utils.data import Dataset, DataLoader
from multiprocessing import Pool

class PostProcess():
    def __init__(self, num_workers=16, post_func=None) -> None:
        super().__init__()
        self.pool = Pool(num_workers)
        self.ret = []
        self.post_func = post_func

    def __call__(self, outputs):
        for r in self.ret:
            pass
        self.ret = self.pool.imap_unordered(self.post_func, outputs)
    
    def join(self):
        for r in self.ret:
            pass
        
