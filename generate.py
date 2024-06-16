import os
import torch
from torchvision import transforms
from models.diffusion import DiffusionModel
from models.unet import Unet
from utils.networkHelper import num_to_groups

"""
Notice : 需要将创建一个名为 "{}_dimMults{}_w{}_p{}_schedule{}_timesteps{}" 的文件夹，放于 "./ckpt/" 中
         用于存放 BestModel.pth , 其中dimMults参数必须与模型训练时的设置一致，其余参数可以自行调整
         例如：attention_dimMults(1, 2, 2)_w4_p0.1_schedulecosine_beta_schedule_timesteps1000
"""

# 生成参数
attention = False  # False : Using MLP replace attention block
transform = True
sample_batch_size = 64
var_schedule = "cosine_beta_schedule"  # 四种方差生成策略，具体见"varianceSchedule.py"
timesteps = 1000
image_size = 64
channels = 1
num_labels = 16
dim_mults = (1, 2, 2,)  # TODO 必须与模型训练时的设置一致
w = 4  # 条件强度
p = 0.1  # 训练时以0.1的概率使用无标签训练(与生成无关但为了标记模型)
num_images = 2048  # 生成图像数量


if __name__ == '__main__':
    # 初始化模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    denoise_model = Unet(dim=image_size,
                         cond_dim=num_labels,
                         channels=channels,
                         dim_mults=dim_mults)

    model = DiffusionModel(var_schedule=var_schedule,
                           timesteps=timesteps,
                           beta_start=0.0001,
                           beta_end=0.02,
                           num_labels=num_labels,
                           device=device,
                           denoise_model=denoise_model).to(device)

    block = 'attention' if attention else 'mlp'
    setting = "{}_dimMults{}_w{}_p{}_schedule{}_timesteps{}".format(block,dim_mults, w, p, var_schedule,
                                                                 timesteps)

    # 加载模型
    model_dir = os.path.join("./ckpt", setting)
    best_model_path = model_dir + '/' + 'BestModel.pth'
    model.load_state_dict(torch.load(best_model_path))

    # 图像保存地址
    generated_images_dir = model_dir + '/' + 'generated_dataset'
    os.makedirs(generated_images_dir, exist_ok=True)

    model.eval()
    image_count = 0
    batches = num_to_groups(num_images, sample_batch_size)
    with torch.no_grad():
        for i, batch_size in enumerate(batches):
            if image_count >= num_images:
                break
            labels = torch.randint(0, num_labels-1, size=(batch_size,), device=device)

            samples = model(mode="infer", img_size=image_size, batch_size=batch_size,
                            channels=channels, c=labels, w=w)
            sample = samples[-1]
            for j, image in enumerate(sample):
                save_path = os.path.join(generated_images_dir,
                                         f'image_{image_count}_label_{labels[j]}.png')
                if transform:
                    # 逆归一化
                    inverse_transform = transforms.Compose([
                        transforms.Normalize(mean=[-1], std=[2])  # 逆归一化公式
                    ])
                    sample = inverse_transform(image)
                image_pil = transforms.ToPILImage()(image)
                image_pil.save(save_path)
                image_count += 1
                if image_count >= num_images:
                    break
