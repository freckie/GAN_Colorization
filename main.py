# 참고 : https://github.com/taeoh-kim/Pytorch_Pix2Pix

import torch
import logging
import torchvision
import torch.nn as nn
import torchvision.utils
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data.dataloader

from PIL import Image
from torch.autograd import Variable
from torchvision import datasets, transforms

from mydataset import MyDataset
from network import Generator as Net_G
from network import Discriminator as Net_D


def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)


def GAN_Loss(input_, is_real, criterion):
    # 진짜 샘플 Loss 계산
    if is_real:
        tmp_tensor = torch.FloatTensor(input_.size()).fill_(1.0)
        rgbs = Variable(tmp_tensor, requires_grad=False)
    # 가짜 샘플 Loss 계산
    else:
        tmp_tensor = torch.FloatTensor(input_.size()).fill_(0.0)
        rgbs = Variable(tmp_tensor, requires_grad=False)
    
    return criterion(input_, rgbs)


if __name__ == '__main__':
    # logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('[%(asctime)s] %(message)s')
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(formatter)
    logger.addHandler(streamHandler)

    # Data Loader
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = MyDataset('./data/processed', transform=transform)
    data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=10, shuffle=True)
    logger.info('데이터 로드 완료.')

    # 모델 생성
    model_G = Net_G()
    model_D = Net_D()

    # Criterion, Optimizer
    criterion = nn.BCELoss()
    criterionL1 = nn.L1Loss()
    optim_G = optim.Adam(model_G.parameters())
    optim_D = optim.Adam(model_G.parameters())

    # 학습
    epochs = 10
    for epoch in range(epochs):
        for idx, (tensor_grey, tensor_rgb) in enumerate(data_loader):
            # ==== D 학습 ==== #
            model_D.zero_grad()

            # 변수
            grey = Variable(tensor_grey)
            real_rgb = Variable(tensor_rgb)
            # 가짜 RGB 생성
            fake_rgb = model_G(grey)

            # 예측 후 Loss 계산 (가짜 샘플)
            pred_fake = model_D(grey, fake_rgb)
            loss_D_fake = GAN_Loss(pred_fake, False, criterion)
            # 예측 후 Loss 계산 (진짜 샘플)
            pred_real = model_D(grey, real_rgb)
            loss_D_real = GAN_Loss(pred_fake, True, criterion)

            # Loss 합산 후 오류 역전파
            loss_D = (loss_D_fake + loss_D_real) * 0.5
            loss_D.backward(retain_graph=True)
            optim_D.step()

            # ==== G 학습 ==== #
            model_G.zero_grad()

            # 예측 후 Loss 계산
            pred_fake = model_D(grey, fake_rgb)
            loss_G_GAN = GAN_Loss(pred_fake, True, criterion)
            loss_G_L1 = criterionL1(fake_rgb, real_rgb)

            # Loss 합산 후 오류 역전파
            loss_G = loss_G_GAN + loss_G_L1 * 100
            loss_G.backward()
            optim_G.step()

            # ==== 학습 로그 ==== #
            logger.info('Epoch [{}/{}] IDX [{}/{}] D_Real_Loss: {:.6f}, D_Fake_Loss: {:.6f}, G_Loss: {:.6f}, G_L1_Loss: {:.6f}'.format(
                epoch + 1, epochs, idx + 1, len(data_loader), loss_D_real.data[0], loss_D_fake.data[0], loss_G_GAN.data[0], loss_G_L1.data[0]))

            # IDX 10번마다 생성 모델 저장 및 샘플 생성
            if idx % 10 == 0:
                torch.save(model_G.state_dict(), './generator.pth')
                # G(grey)
                temp = model_G(tensor_grey)
                torchvision.utils.save_image(denorm(temp), './result/{}_{}_fake.jpg'.format(epoch + 1, idx + 1))
                torchvision.utils.save_image(denorm(tensor_rgb), './result/{}_{}_real.jpg'.format(epoch + 1, idx + 1))