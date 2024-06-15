import torch

from utils.networkHelper import *
from models.varianceSchedule import VarianceSchedule

class DiffusionModel(nn.Module):
    def __init__(self,
                 var_schedule="linear_beta_schedule",
                 timesteps=1000,
                 beta_start=0.0001,
                 beta_end=0.02,
                 device='cpu',
                 denoise_model=None
                 ):
        super().__init__()
        self.net = denoise_model
        self.device = device
        var_schedule_func = VarianceSchedule(var_schedule=var_schedule,
                                             beta_start=beta_start,
                                             beta_end=beta_end)
        self.timesteps = timesteps
        self.betas = var_schedule_func(timesteps)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        # 采样方差使用DDPM论文中的方差下界
        self.sample_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)


    def q_sample(self, x_0, t, noise=None):
        # 前向扩散过程采样x_t
        if noise is None:
            noise = torch.randn_like(x_0)

        alphas_cumprod_t = extract(self.alphas_cumprod, t, x_0.shape)
        sqrt_alphas_cumprod_t = torch.sqrt(alphas_cumprod_t)
        sqrt_one_minus_alphas_cumprod_t = torch.sqrt(1. - sqrt_alphas_cumprod_t)
        return sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise

    def loss_function(self, x_0, t, c, noise=None, loss_type=None):
        if noise is None:
            noise = torch.randn_like(x_0)
        x_t = self.q_sample(x_0=x_0, t=t, noise=noise)
        predicted_noise = self.net(x_t, t, c)

        if loss_type == 'l1':
            loss = F.l1_loss(noise, predicted_noise)
        elif loss_type == 'l2':
            loss = F.mse_loss(noise, predicted_noise)
        else:
            raise NotImplementedError()

        return loss


    @torch.no_grad()
    def p_sample(self, x_t, t, c, w):

        conditional_noise = self.net(x_t, t, c)
        cnone = torch.ones_like(c) * 16
        unconditional_noise = self.net(x_t, t, cnone)
        predicted_noise = (1+w) * conditional_noise - w * unconditional_noise

        betas_t = extract(self.betas, t, x_t.shape)
        alphas_t = extract(self.alphas, t, x_t.shape)
        sqrt_recip_alphas_t = torch.sqrt(1.0 / alphas_t)
        alphas_cumprod_t = extract(self.alphas_cumprod, t, x_t.shape)
        sqrt_one_minus_alphas_cumprod_t = torch.sqrt(1. - alphas_cumprod_t)

        mean = (x_t - (predicted_noise * betas_t / sqrt_one_minus_alphas_cumprod_t)) * sqrt_recip_alphas_t

        if t[0] == 0:
            return mean
        else:
            sample_variance_t = extract(self.sample_variance, t, x_t.shape)
            noise = torch.randn_like(x_t)
            return mean + torch.sqrt(sample_variance_t) * noise

    def forward(self, mode, **kwargs):
        if mode == "train":
            """
            使用 https://arxiv.org/pdf/2207.12598 中 Algorithm 1 训练
            以概率 p_unconditional 随机丢弃条件c
            """
            required_keys = {"x_0", "t", "c"}
            if required_keys.issubset(kwargs.keys()):
                x_0 = kwargs["x_0"]
                t = kwargs["t"]
                c = kwargs["c"]
            else:
                raise ValueError("x_0 and t must be specified for training.")

            noise = kwargs.get("noise", None)
            loss_type = kwargs.get("loss_type", "l2")
            p_unconditional = kwargs.get("p_unconditional", 0.1)
            c_mask = torch.bernoulli(p_unconditional * torch.ones_like(c))
            c_copy = c.clone()  # 对c的副本进行操作，防止修改原始c

            for i in range(c_mask.shape[0]):
                if c_mask[i] == 1:
                    c_copy[i] = 16

            c_copy = c_copy.long().to(self.device)

            return self.loss_function(x_0=x_0, t=t, c=c_copy, noise=noise, loss_type=loss_type)

        elif mode == "infer":
            """
            使用 https://arxiv.org/pdf/2207.12598 中 Algorithm 2 推理
            """
            with torch.no_grad():
                required_keys = {"img_size", "batch_size", "channels", "c"}
                if required_keys.issubset(kwargs.keys()):
                    img_size = kwargs["img_size"]
                    batch_size = kwargs["batch_size"]
                    channels = kwargs["channels"]
                    c = kwargs["c"]
                else:
                    raise ValueError("img_size , batch_size , channels must be specified for generating.")

                w = kwargs.get("w", 0.)

                img = torch.randn((batch_size, channels, img_size, img_size), device=self.device)
                imgs = []

                for i in tqdm(reversed(range(0, self.timesteps)), desc='Sampling loop', total=self.timesteps):
                    img = self.p_sample(img, torch.full((batch_size,), i, device=self.device, dtype=torch.long), c, w)
                    imgs.append(img.cpu().numpy())
                return imgs
        else:
            raise ValueError("mode must be 'train' or 'infer'")
