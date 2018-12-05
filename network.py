import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from PIL import Image
from torch.autograd import Variable
from torchvision import datasets, transforms


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(1 + 3, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),

            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),

            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=1),
            nn.BatchNorm2d(512),

            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x_grey, x_rgb):
        out = torch.cat([x_grey, x_rgb], dim=1)
        return self.model(out)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        # (1*256*256) -> (64*128*128)
        self.conv1 = nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1)

        # (64*128*128) -> (128*64*64)
        self.conv2 = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128)
        )

        # (128*64*64) -> (256*32*32)
        self.conv3 = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256)
        )

        # (256*32*32) -> (512*16*16)
        self.conv4 = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512)
        )

        # (512*16*16) -> (512*8*8)
        self.conv5 = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512)
        )

        # (512*8*8) -> (512*4*4)
        self.conv6 = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512)
        )

        # (512*4*4) -> (512*2*2)
        self.conv7 = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512)
        )

        # (512*2*2) -> (512*1*1)
        self.conv8 = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512)
        )

        # (512*1*1) -> (512*2*2)
        self.deconv8 = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(512 * 1, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.Dropout(0.5)
        )

        # ((512*2)*2*2) -> ((512*2)*4*4)
        self.deconv7 = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(512 * 2, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.Dropout(0.5)
        )

        # ((512*2)*4*4) -> ((512*2)*8*8)
        self.deconv6 = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(512 * 2, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.Dropout(0.5)
        )

        # ((512*2)*8*8) -> ((512*2)*16*16)
        self.deconv5 = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(512 * 2, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.Dropout(0.5)
        )

        # ((512*2)*16*16) -> ((256*2)*32*32)
        self.deconv4 = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(512 * 2, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.Dropout(0.5)
        )

        # ((256*2)*32*32) -> ((128*2)*64*64)
        self.deconv3 = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(256 * 2, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.Dropout(0.5)
        )

        # ((128*2)*64*64) -> ((64*2)*128*128)
        self.deconv2 = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(128 * 2, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.Dropout(0.5)
        )

        # ((64*2)*128*128) -> (3*256*256)
        self.deconv1 = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(64 * 2, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        c1 = self.conv1(x)
        c2 = self.conv2(c1)
        c3 = self.conv3(c2)
        c4 = self.conv4(c3)
        c5 = self.conv5(c4)
        c6 = self.conv6(c5)
        c7 = self.conv7(c6)
        c8 = self.conv8(c7)

        d7 = self.deconv8(c8)
        d7 = torch.cat((c7, d7), dim=1)
        d6 = self.deconv7(d7)
        d6 = torch.cat((c6, d6), dim=1)
        d5 = self.deconv6(d6)
        d5 = torch.cat((c5, d5), dim=1)
        d4 = self.deconv5(d5)
        d4 = torch.cat((c4, d4), dim=1)
        d3 = self.deconv4(d4)
        d3 = torch.cat((c3, d3), dim=1)
        d2 = self.deconv3(d3)
        d2 = torch.cat((c2, d2), dim=1)
        d1 = self.deconv2(d2)
        d1 = torch.cat((c1, d1), dim=1)
        out = self.deconv1(d1)

        return out