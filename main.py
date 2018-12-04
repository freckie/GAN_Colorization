import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data.dataloader

from PIL import Image
from torch.autograd import Variable
from torchvision import datasets, transforms

from mydataset import MyDataset
from network import Generator as Net_G
from network import Discriminator as Net_D


if __name__ == '__main__':
    # Data Loader
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = MyDataset('./data/processed', transform=transform)
    data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=100, shuffle=True)

    # 모델 생성
    model_G = Net_G()
    model_D = Net_D()

    # Criterion, Optimizer
    criterion = nn.BCELoss()
    optim_G = optim.Adam(model_G.parameters())
    optim_D = optim.Adam(model_G.parameters())

    # 학습
    epochs = 100
    for epoch in range(epochs):
        data_loader = list()
        for batch_idx, (grey, rgb) in enumerate(data_loader):
            pass