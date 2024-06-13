import torch
from torch import nn
import math
import torch.nn.functional as F
from tqdm.auto import tqdm

def extract(tensor, idx, x_shape):
    # 从tensor最后一维中索引idx对应元素
    batch_size = idx.shape[0]
    out = tensor.gather(-1, idx.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(idx.device)