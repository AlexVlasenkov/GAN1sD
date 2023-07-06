import numpy as np
from PIL import Image
arrayNumber = np.zeros((512, 512))
arraySource = []
f1 = open('./test/images/wgan_epoch_10711_.txt')
for line in f1:
      arraySource.append(line.strip())
for i in range(512):
    for j in range(512):
        arrayNumber[i][j] = int(arraySource[j])
arrayNumber = arrayNumber.transpose()
img = Image.fromarray(arrayNumber )
img.show()



