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
        img_rgb = Image.open(self.path + '/{}_rgb.jpg'.format(index))
        img_grey = Image.open(self.path + '/{}_grey.jpg'.format(index))
        tensor_rgb = self.transform(img_rgb)
        tensor_grey = self.transform(img_grey)
        # tensor_rgb = self.to_tensor(img_rgb)
        # tensor_grey = self.to_tensor(img_grey)

        return (tensor_grey, tensor_rgb)

    def __len__(self):
        return len(self.img_filenames)
