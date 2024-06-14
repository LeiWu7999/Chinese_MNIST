import os
import numpy as np
import torchvision
import torch
import torchvision.transforms as transforms
from torch.optim import Adam
from torch.utils.data import DataLoader

from utils.networkHelper import *

from models.diffusion import DiffusionModel
from models.unet import Unet
from utils.dataloader import Chinese_MNIST
from utils.trainHelper import SimpleDiffusionTrainer

root_dir = os.path.abspath(os.path.dirname(__file__))
image_dir = os.path.join(root_dir, 'dataset\\data')

img_size = 64
channels = 1
batch_size = 64
num_labels = 16
timesteps = 1000
lr = 1e-3
epoches = 20
transform = False
var_scheduler = "linear_beta_schedule"

dataset = Chinese_MNIST(root_dir, transform=transform)
dataloader = DataLoader(dataset,
                        batch_size=batch_size,
                        shuffle=True,
                        num_workers=1)
device = "cuda" if torch.cuda.is_available() else "cpu"
dim_mults = (1, 2, 4, 8)

denoise_model = Unet(dim=img_size,
                     cond_dim=num_labels,
                     channels=channels,
                     dim_mults=dim_mults)

Model = DiffusionModel(var_schedule=var_scheduler,
                       timesteps=timesteps,
                       beta_start=0.0001,
                       beta_end=0.02,
                       device=device,
                       denoise_model=denoise_model).to(device)

optimizer = Adam(Model.parameters(), lr=lr)

Trainer = SimpleDiffusionTrainer(epoches=epoches,
                                 train_loader=dataloader,
                                 optimizer=optimizer,
                                 device=device,
                                 timesteps=timesteps)

root_path = "./ckpt"
setting = "imageSize{}_channels{}_dimMults{}_timeSteps{}_scheduleName{}".format(img_size, channels, dim_mults, timesteps, var_scheduler)

saved_path = os.path.join(root_path, setting)
if not os.path.exists(saved_path):
    os.makedirs(saved_path)

# 训练好的模型加载，如果模型是已经训练好的，则可以将下面两行代码取消注释
# best_model_path = saved_path + '/' + 'BestModel.pth'
# Model.load_state_dict(torch.load(best_model_path))

# 如果模型已经训练好则注释下面这行代码，反之则注释上面两行代码
Model = Trainer.train(Model, model_save_path=saved_path)

labels = torch.randint(0, num_labels, size=(64,))
c = torch.zeros((len(labels), num_labels))
for i, label in enumerate(labels):
    c[i][label] = 1

samples = Model(mode="infer", img_size=img_size, batch_size=64, channels=channels, c=c)

# 随机挑一张显示
random_index = 1
generated_image = samples[-1][random_index].reshape(channels, img_size, img_size)

if transform:
    # 逆归一化
    inverse_transform = transforms.Compose([
        transforms.Normalize(mean=[-1], std=[2])  # 逆归一化公式
    ])
    generated_image = inverse_transform(torch.from_numpy(generated_image))

# 转换为 PIL 图像
to_pil = transforms.ToPILImage()
generated_image_pil = to_pil(generated_image)

generated_image_pil.show()



