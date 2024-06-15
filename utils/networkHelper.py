import torch
from torch import nn
import math
import torch.nn.functional as F
from tqdm.auto import tqdm
from inspect import isfunction
from einops.layers.torch import Rearrange
from torchvision.transforms import Compose, Lambda, ToPILImage
import IPython
e = IPython.embed

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

def default(val, d):
    """
    该函数的目的是提供一个简单的机制来获取给定变量的默认值。
    如果 val 存在，则返回该值。如果不存在，则使用 d 函数提供的默认值，
    或者如果 d 不是一个函数，则返回 d。
    :param val:需要判断的变量
    :param d:提供默认值的变量或函数
    :return:
    """
    if exists(val):
        return val
    return d() if isfunction(d) else d

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        sin_embeddings = embeddings.sin()
        cos_embeddings = embeddings.cos()
        interleaved_embeddings = torch.zeros((embeddings.size(0), self.dim), device=device)
        interleaved_embeddings[:, 0::2] = sin_embeddings
        interleaved_embeddings[:, 1::2] = cos_embeddings

        # embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)

        return interleaved_embeddings


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


def Downsample(dim_in, dim_out=None):
    return nn.Sequential(
        Rearrange("b c (h p1) (w p2) -> b (c p1 p2) h w", p1=2, p2=2),
        nn.Conv2d(dim_in * 4, default(dim_out, dim_in), kernel_size=1),
    )

def Upsample(dim_in, dim_out=None):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.Conv2d(dim_in, default(dim_out, dim_in), 3, padding=1),
    )

