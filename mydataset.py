import os
from PIL import Image
from torch.utils import data
from torchvision import transforms

class MyDataset(data.Dataset):

    def __init__(self, path, transform=None):
        self.to_tensor = transforms.ToTensor()
        self.path = path
        self.transform = transform
        self.img_filenames = os.listdir(path)

    def __getitem__(self, index):
        img_rgb = Image.open(self.path + '/{}_rgb.jpg'.format(index)).convert('RGB')
        img_grey = Image.open(self.path + '/{}_grey.jpg'.format(index)).convert('L')
        tensor_rgb = self.transform(img_rgb)
        tensor_grey = self.transform(img_grey)
        img_rgb.close()
        img_grey.close()

        return (tensor_grey, tensor_rgb)

    def __len__(self):
        return int(len(self.img_filenames) / 2)
