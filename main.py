import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from PIL import Image
from torch.autograd import Variable
from torchvision import datasets, transforms

from network import Generator as Net_G
from network import Discriminator as Net_D


if __name__ == '__main__':
    # Data Loader

    # 모델 생성
    model_G = Net_G()
    model_D = Net_D()

    # Criterion, Optimizer
    criterion = nn.BCELoss()
    optim_G = optim.Adam(model_G.parameters())
    optim_D = optim.Adam(model_G.parameters())

    