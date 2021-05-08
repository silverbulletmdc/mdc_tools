from IPython import get_ipython

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

import numpy as np
import cv2
import torch
from torch import nn
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm