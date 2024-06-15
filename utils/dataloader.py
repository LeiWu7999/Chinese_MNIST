import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import IPython
e = IPython.embed


class Chinese_MNIST(Dataset):
    def __init__(self, root_dir, transform=False):
        self.root_dir = root_dir
        self.image_paths = []
        self.labels = []
        self.transform = transforms.Compose([
                transforms.Normalize((0.5,), (0.5,))
            ]) if transform else None
        # Dataset shape is (100, 10, 15)
        data_dir = os.path.join(root_dir, 'dataset/data')
        for img_name in os.listdir(data_dir):
            img_path = os.path.join(data_dir, img_name)
            self.image_paths.append(img_path)
            label = int(img_name.split('_')[-1].split('.')[0]) - 1
            self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = transforms.ToTensor()(Image.open(img_path))
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


# if __name__ == "__main__":
#     root_dir = 'C:\\Users\\admin\\Desktop\\Chinese_MNIST'  # 替换为你的数据集路径
#     dataset = Chinese_MNIST(root_dir=root_dir, transform=False)
#     dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=1)
#     for images, labels in dataloader:
#         print(images.shape)  # 打印批次图像的形状
#         print(labels)  # 打印批次标签
#         break  # 仅测试一个批次
