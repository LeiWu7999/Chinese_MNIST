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

def exists(x):
    """
    判断数值是否为空
    :param x: 输入数据
    :return: 如果不为空则True 反之则返回False
    """
    return x is not None