import torch
import torch.nn as nn
import torchvision.datasets as dataset
import torch.optim as optim
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

from wganImage import Discriminator, Generator, weights_init
from preprocessingImage import Dataset
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset,DataLoader
import io
from ra import calculation_Ra

n_critic = 5
clip_value = 0.01
lr = 1e-4
epoch_num = 150
batch_size = 8
nz = 100  # length of noise
ngpu = 0
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def calcIndexGenArray(arrayGen, arrayBright, arrayFake):
    raznOld = 100000
    arrayNumber = []
    arrayFakeNum = arrayFake.numpy()
    arrayFakeNumList = arrayFakeNum.tolist()
    for elem in range(0, 512):
        arrayNumber.append(elem)
    #plt.plot(arrayNumber, arrayFakeNumList)
    for elem in range(len(arrayFakeNumList)):
        raznOld = 100000
        raznZeroOld = 100000
        for elemGen in range(256):
            if (arrayFakeNumList[elem] > 0 ):
                razn = arrayFakeNumList[elem] - arrayGen[elemGen]
                raznZero = 0 + abs(razn)
                if ((razn < raznOld and razn > 0) or raznZero < raznZeroOld):
                    if (raznZero < raznZeroOld):
                        raznZeroOld = raznZero
                        index = elemGen
                        if (razn < raznOld and razn > 0):
                            raznOld = razn
                    else:
                        raznOld = razn
                        index = elemGen
            else:
                razn =  arrayFakeNumList[elem] + abs(arrayGen[elemGen])
                raznZero = 0 + abs(razn)
                if ((razn < raznOld and razn > 0) or (raznZero < raznZeroOld and razn > 0)):
                    if (raznZero < raznZeroOld):
                        raznZeroOld = raznZero
                        index = elemGen
                        if (razn < raznOld and razn > 0):
                            raznOld = razn
                    else:
                        raznOld = razn
                        index = elemGen
        arrayFakeNumList[elem] = arrayBright[index]
    #plt.plot(arrayNumber, arrayFakeNumList)
    return arrayFakeNumList



def main():
    # load training data
    numelem = 2592
    img = Image.open('data/Image/img1.jpg')
    img_gray = img.convert('L')

    arrayNumber = []
    for elem in range(0,512):
        arrayNumber.append(elem)
    arrayBright = []
    for elem in range(0,256):
        arrayBright.append(elem)
    arrayGen = []
    temp = -1
    for elem in range(0, 256):
        arrayGen.append(temp)
        temp+=0.0078125

    imgByteArr1 = np.asarray(img_gray)
    imgByteArrString = imgByteArr1[:,0]
    imgByteArrStringCut = imgByteArrString[:512]
    #f = open('analys/image/processFile.txt', 'w')
    #for i in range(imgByteArrStringCut.__len__()):
        #f.writelines(str(imgByteArrStringCut[i])+'\n')
    #f.close()
    plt.plot(arrayNumber,imgByteArrStringCut)
    plt.show()
    imgByteArr2 = imgByteArr1.flatten()

    dataset = []
    #trainset = Dataset('data/brilliant_blue')
    #tensor_x = torch.Tensor(imgByteArr2)
    #a = np.reshape(imgByteArr2, (imgByteArr2.size, 1))
    a = np.reshape(imgByteArrStringCut, (imgByteArrStringCut.size, 1))
    trainset = Dataset(a)
    #trainset = Dataset(imgByteArr2)
    #tensor_x = torch.Tensor(imgByteArr2)
    #my_dataset = TensorDataset(tensor_x)
   # dataset = np.vstack(imgByteArr2).T
   # dataset = torch.from_numpy(dataset).float()
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
    fakeRa.append(calculation_Ra(arrayNumber, imgByteArrStringCut))
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

        # save training process
        with torch.no_grad():
            fake = netG(fixed_noise).detach().cpu()
            #f, a = plt.subplots(4, 4, figsize=(8, 8))
            # f, a = plt.subplots(4, 4, figsize=(8, 8))
            '''for i in range(4):
                for j in range(4):'''
            for i in range(16):
                plt.figure()
                arrayFake = calcIndexGenArray(arrayGen, arrayBright,fake[i].view(-1))
                #plt.plot(fake[i].view(-1))
                plt.plot(arrayFake)
                np.savetxt('./imageAlone/txtfiles/wgan_epoch_' + str(epoch) +'_'+ str(i) + '_' + '.txt', arrayFake)
                #fakeRa.append(calculation_Ra(arrayNumber, arrayFake))
                plt.savefig('./imageAlone/wgan_epoch_' + str(epoch) + '_' + str(i) + '.png')
                plt.close()
            ''' for i in range(4):
                for j in range(4):

                    a[i][j].plot(fake[i * 4 + j].view(-1))
                    arrayFake = calcIndexGenArray(arrayGen, arrayBright, fake[i * 4 + j].view(-1))
                    fakeRa.append(calculation_Ra(arrayNumber, arrayFake))
                    np.savetxt('./test/images/wgan_epoch_' + str(epoch) + str(i*4+j) + '_' + '.txt', arrayFake, fmt='%d')
                    a[i][j].set_xticks(())
                    a[i][j].set_yticks(())
                    a[i][j].set_xlabel('MSE Error: {}'.format(mean_squared_error(imgByteArrStringCut, arrayFake)))
                    plt.savefig('./img/wgan_epoch_%d.png' % epoch)
            plt.close()'''
    # save model
    torch.save(netG, './nets/wgan_netG.pkl')
    torch.save(netD, './nets/wgan_netD.pkl')


if __name__ == '__main__':
    main()
