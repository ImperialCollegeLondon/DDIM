from __future__ import print_function

import sys
import os
import argparse
import random
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
from torch.autograd import Variable
from dataset import HDF5Dataset
from hdf5_io import save_hdf5
import WGANFPNX
import numpy as np
import time
import argparse

def main():
    time_start = time.time()
    torch.cuda.set_device(0)
    print('start time is', time_start)

    # Change workdir to where you want the files output
    work_dir = os.path.expandvars('Multi/')

    # 命令行参数解析
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, help='3D')


    parser.add_argument('--dataroot', required=True, help='path to dataset')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
    parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
    parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
    parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
    parser.add_argument('--ngf', type=int, default=64)
    parser.add_argument('--ndf', type=int, default=64)
    parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--cuda', action='store_true', help='enables cuda')
    parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
    parser.add_argument('--netG', default='', help="path to netG (to continue training)")
    parser.add_argument('--netD', default='', help="path to netD (to continue training)")
    parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints')
    parser.add_argument('--weight_clip_value', type=float, default=0.01, help='weight clip value')
    parser.add_argument('--gradient_penalty_weight', type=float, default=5, help='gradient_penalty')
    parser.add_argument('--weight_decay', type=float, default=0.001, help='weight_decay')

    opt = parser.parse_args()
    print(opt)

    try:
        os.makedirs(opt.outf)
    except OSError:
        pass

    # 移除随机种子设置

    cudnn.benchmark = True

    # 检查CUDA设备是否可用
    if torch.cuda.is_available() and not opt.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    # 移除输入数据集相关代码
    output_dataset = HDF5Dataset(os.path.join(opt.dataroot, 'wet25564'),
                                 input_transform=transforms.Compose([
                                     transforms.ToTensor()
                                 ]))

    output_dataloader = torch.utils.data.DataLoader(output_dataset, batch_size=opt.batchSize,
                                                    shuffle=True, num_workers=int(opt.workers),
                                                    pin_memory=True)
    ngpu = int(opt.ngpu)
    nz = int(opt.nz)
    ngf = int(opt.ngf)
    ndf = int(opt.ndf)
    weight_decay = opt.weight_decay
    weight_clip_value = opt.weight_clip_value
    weight_gradient_penalty_weight = opt.gradient_penalty_weight
    nc = 1  # 确保这个值匹配您的输入数据的通道数


    # Custom weights initialization
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)


    device = torch.device("cuda" if opt.cuda else "cpu")
    n_levels = 4
    netG = WGANFPNX.LaplacianPyramidGenerator(opt.imageSize, nz, nc, ngf, ngpu).to(device)
    netG.apply(weights_init)
    if opt.netG != '':
        netG.load_state_dict(torch.load(opt.netG))
    #print(netG)

    netD = WGANFPNX.WGAN3D_D(opt.imageSize, nz, nc, ndf, ngpu).to(device)
    netD.apply(weights_init)
    if opt.netD != '':
        netD.load_state_dict(torch.load(opt.netD))
    #print(netD)

    wasserstein_criterion = WGANFPNX.WassersteinLP(opt.weight_clip_value, weight_gradient_penalty_weight, netD)

    label = torch.FloatTensor(opt.batchSize).to(device)
    real_label = 1
    fake_label = 0

    label = Variable(label)

    # Setup optimizer
    optimizerD = optim.RMSprop(netD.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
    optimizerG = optim.RMSprop(netG.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)

    gen_iterations = 0
    dis_iterations = 0
    f = open(work_dir + "training_curve.csv", "a")
    for epoch in range(opt.niter):
        for i, real_output in enumerate(output_dataloader, 0):
            # 移动数据到设备
            real_output = real_output.to(device)

            ############################
            # (1) 更新判别器网络：最大化 log(D(x)) + log(1 - D(G(z)))
            ###########################
            netD.zero_grad()

            # 训练真实数据
            output = netD(real_output)
            errD_real = -output.mean()
            errD_real.backward()

            # 训练生成的数据
            random_input = torch.randn(opt.batchSize, nz, 1, 1, 1, device=device)  # 生成随机噪声作为生成器输入
            fake_output = None
            for level in range(n_levels):
                if level == 0:
                    fake_output = netG(random_input, level)
                else:
                    fake_output = netG(random_input, level, fake_output)
            output = netD(fake_output.detach())
            errD_fake = output.mean()
            errD_fake.backward()

            errD = errD_real - errD_fake
            optimizerD.step()

            ############################
            # (2) 更新生成器网络：最大化 log(D(G(z)))
            ###########################
            netG.zero_grad()
            fake_output = None
            for level in range(n_levels):
                if level == 0:
                    fake_output = netG(random_input, level)
                else:
                    fake_output = netG(random_input, level, fake_output)
            output = netD(fake_output)
            errG = -output.mean()
            errG.backward()
            optimizerG.step()

            print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f'
                  % (epoch, opt.niter, i, len(output_dataloader), errD.item(), errG.item()))
            f.write('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f\n'
                    % (epoch, opt.niter, i, len(output_dataloader), errD.item(), errG.item()))
            f.write('\n')

    # Generate and save fake samples after each epoch
        if (epoch + 1) % 1 == 0:
            random_noise = torch.randn(opt.batchSize, nz, 1, 1, 1, device=device)
            for level in range(n_levels):
                if level == 0:
                    fake = netG(random_noise, level)
                else:
                    fake = netG(random_noise, level, fake)
            save_hdf5(fake.data, work_dir + 'fake_samples_epoch_{0}.hdf5'.format(epoch + 1))

    # Do checkpointing
        if (epoch + 1) % 1 == 0:
            torch.save(netG.state_dict(), work_dir + 'netG_epoch_%d.pth' % (epoch + 1))
            torch.save(netD.state_dict(), work_dir + 'netD_epoch_%d.pth' % (epoch + 1))
    f.close()
    time_end = time.time()
    print('end time is', time_end)
    print('total time cost is', time_end - time_start)

if __name__ == "__main__":
    main()
