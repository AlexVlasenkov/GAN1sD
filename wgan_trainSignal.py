import torch
import torch.nn as nn
import torchvision.datasets as dataset
import torch.optim as optim
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

from ra import calculation_Ra
from wganSignal import Discriminator, Generator, weights_init
from preprocessingSignal import Dataset
from PIL import Image
from torch.utils.data import TensorDataset,DataLoader
import io


n_critic = 5
#clip_value = 0.01
clip_value = 0.01
#lr = 1e-4
lr = 0.0001
epoch_num = 200
batch_size = 8
nz = 512  # length of noise
ngpu = 0
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")




def main():
    # load training data
    #img = Image.open('data/Image/img1(100).jpg')
    #img_gray = img.convert('L')
    #img_gray.show()
    #imgByteArr1 = np.asarray(img_gray)
   #imgByteArr2 = imgByteArr1.flatten()
    #dataset = []
    trainset = Dataset('data/brilliant_blue')


    #tensor_x = torch.Tensor(imgByteArr2)
    #a = np.reshape(imgByteArr2, (imgByteArr2.size, 1))
    #trainset = Dataset(a)
    #trainset = Dataset(imgByteArr2)
    #tensor_x = torch.Tensor(imgByteArr2)
    #my_dataset = TensorDataset(tensor_x)
   # dataset = np.vstack(imgByteArr2).T
   # dataset = torch.from_numpy(dataset).float()
    arraySource = []
    f1 = open('data/brilliant_blue/array.txt')
    for line in f1:
        ind = line.find(',')
        line2 = line[ind + 1:len(line)]
        arraySource.append(float(line2.strip()))

    numelem = 512
    arrayNumber = []
    for elem in range(0, numelem):
        arrayNumber.append(elem)
    ra = calculation_Ra(arrayNumber, arraySource)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True
    )
    
    # init netD and netG
    netD = Discriminator().to(device)
    netD.apply(weights_init)

    netG = Generator(nz).to(device)
    netG.apply(weights_init)

    # used for visualizing training process
    fixed_noise = torch.randn(16, nz, 1, device=device)
    # optimizers
    optimizerD = optim.RMSprop(netD.parameters(), lr=lr)
    optimizerG = optim.RMSprop(netG.parameters(), lr=lr)
    fakeRa = []
    for epoch in range(epoch_num):
        for step, (data, _) in enumerate(trainloader):
            # training netD
            real_cpu = data.to(device)
            b_size = real_cpu.size(0)
            netD.zero_grad()

            noise = torch.randn(b_size, nz, 1, device=device)
            fake = netG(noise)

            loss_D = -torch.mean(netD(real_cpu)) + torch.mean(netD(fake))

            loss_D.backward()
            optimizerD.step()

            for p in netD.parameters():
                p.data.clamp_(-clip_value, clip_value)

            if step % n_critic == 0:
                # training netG
                noise = torch.randn(b_size, nz, 1, device=device)

                netG.zero_grad()
                fake = netG(noise)
                loss_G = -torch.mean(netD(fake))

                netD.zero_grad()
                netG.zero_grad()
                loss_G.backward()
                optimizerG.step()
            
            if step % 5 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f'
                    % (epoch, epoch_num, step, len(trainloader), loss_D.item(), loss_G.item()))



        with torch.no_grad():
            fake = netG(fixed_noise).detach().cpu()
            #f, a = plt.subplots(4, 4, figsize=(8, 8))
            #for i in range(4):
                #for j in range(4):
            fakeArr = []

            # save training process

            #for i in range(4):
                #for j in range(4):

            for i in range(16):
                plt.figure()
                plt.plot(fake[i].view(-1))
                plt.savefig('./test/wgan_epoch_'+str(epoch)+str(i)+'_'+'.png')
                plt.close()
                np.savetxt('./test/txtfiles/wgan_epoch_'+str(epoch)+str(i)+'_'+'.txt',fake[i].view(-1))
                #fakeArr.append(fake[i].cpu().data.numpy())
                #fakeArr = np.reshape(fakeArr, (512))
                #fakeRa.append(mean_squared_error(arrayNumber, fakeArr))
                #fakeArr = []
                #plt.plot(fake.view(-1))
                #a[i][j].plot(fake[i * 4 + j].view(-1))
                #a[i][j].set_xticks(())
                #a[i][j].set_yticks(())
            #plt.savefig('./img/wgan_epoch_%d.png' % epoch)
            #plt.close()

        '''  
    fakeArr = []
    fakeRa = []
    # save training process
    for i in range(16):
        fakeArr.append(fake[i].cpu().data.numpy())
        fakeArr = np.reshape(fakeArr,(512))
        fakeRa.append(calculation_Ra(arrayNumber, fakeArr))
        fakeArr=[]
    #save model'''

    torch.save(netG, './nets/wgan_netG.pkl')
    torch.save(netD, './nets/wgan_netD.pkl')


if __name__ == '__main__':
    main()
