import os
import argparse
import torchvision.transforms as transforms
from torchvision.utils import save_image
from matplotlib import pyplot as plt
from torch.optim import Adam
from torch.utils.data import DataLoader
from pathlib import Path

from utils.networkHelper import *
from models.diffusion import DiffusionModel
from models.unet import Unet
from utils.dataloader import Chinese_MNIST
from utils.trainHelper import SimpleDiffusionTrainer


def main(args):
    root_dir = os.path.abspath(os.path.dirname(__file__))

    """
    -------------------------------------------Tuning hyperparameters here----------------------------------------------
    """
    attention = False  # False : Using MLP replace attention block
    img_size = 64
    channels = 1
    batch_size = 64
    sample_batch_size = 32
    num_labels = 16  # 总共15个字，外加一个表示无标签的情况
    timesteps = 1000  # 采样步数
    lr = 1e-3   # 学习率
    loss_type = "l2"  # l1 or l2 or huber
    w = 4  # 条件强度，w越大图像越贴合标签，但多样性降低
    p_unconditional = 0.1  # 训练时以0.1的概率使用无标签训练
    epoches = 20
    transform = True  # 是否对图像进行预处理
    var_scheduler = "cosine_beta_schedule"  # 扩散过程设方差设置策略
    dim_mults = (1, 2, 2,)  # U-Net 上下采样缩放比例，如降采样阶段：(1,2,4,8)->图像尺寸:(不变,减半,减四倍,减八倍)
    save_and_sample_every = 1000  # 训练过程中每过多少步采样一次图片
    """
    -------------------------------------------Tuning hyperparameters above---------------------------------------------
    """

    dataset = Chinese_MNIST(root_dir, transform=transform)
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=1)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    denoise_model = Unet(dim=img_size,
                         cond_dim=num_labels,
                         channels=channels,
                         dim_mults=dim_mults,
                         attention=attention)

    Model = DiffusionModel(var_schedule=var_scheduler,
                           timesteps=timesteps,
                           beta_start=0.0001,
                           beta_end=0.02,
                           num_labels=num_labels,
                           device=device,
                           denoise_model=denoise_model).to(device)

    optimizer = Adam(Model.parameters(), lr=lr)

    Trainer = SimpleDiffusionTrainer(epoches=epoches,
                                     train_loader=dataloader,
                                     optimizer=optimizer,
                                     device=device,
                                     timesteps=timesteps,
                                     save_and_sample_every=save_and_sample_every,
                                     img_size=img_size,
                                     channels=channels,
                                     w=w,
                                     transform=transform,
                                     loss_type=loss_type,
                                     num_labels=num_labels)

    block = 'attention' if attention else 'mlp'
    model_dir = "./ckpt"
    setting = "{}_dimMults{}_w{}_p{}_schedule{}_timesteps{}".format(block,dim_mults,
                                                                 w, p_unconditional, var_scheduler, timesteps)

    saved_path = os.path.join(model_dir, setting)
    if not os.path.exists(saved_path):
        os.makedirs(saved_path)

    if args.mode == 'train':
        Model = Trainer(Model, p_unconditional=p_unconditional, model_save_path=saved_path)

    elif args.mode == 'infer':
        best_model_path = saved_path + '/' + 'BestModel.pth'
        Model.load_state_dict(torch.load(best_model_path))

        labels = torch.randint(0, num_labels-1, size=(sample_batch_size,), dtype=torch.long, device=device)

        samples = Model(mode="infer", img_size=img_size, batch_size=sample_batch_size, channels=channels, c=labels, w=w)
        sample = samples[-1]
        # 随机挑一张显示
        random_index = 1
        generated_image = samples[-1][random_index].numpy().reshape(img_size, img_size, channels)
        plt.imshow(generated_image, cmap='gray')
        plt.show()

        if transform:
            # 逆归一化
            inverse_transform = transforms.Compose([
                transforms.Normalize(mean=[-1], std=[2]),  # 逆归一化公式
                transforms.Resize((img_size, img_size))
            ])
            sample = inverse_transform(sample)

        if args.save:
            save_image(sample, str(Path(saved_path) / f'samples.png'), nrow=6)
            labels_path = str(Path(saved_path) / 'labels.txt')
            with open(labels_path, 'w') as f:
                for label in labels:
                    f.write(f"{label.item()}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-mode", choices=["train", "infer"], required=True,
                        help="Choose whether to train the model or run inference.")
    parser.add_argument("-save", action="store_true", help="Save the generated images during inference.")
    args = parser.parse_args()
    main(args)

