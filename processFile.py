import numpy as np

data = []
# analys/1.txt
with open("array.txt") as f:
    for line in f:
        data.append([float(x) for x in line.split(',')])

arr = np.zeros(data.__len__())
for i in range(data.__len__()):
    arr[i] = data[i][0]
print("a")

# analys/image/wgan_epoch_114_3_.txt
res = np.loadtxt("brilliant_blue.txt", delimiter='\t', dtype=np.cfloat)

# analys/image/processFile3.txt
f = open('output.txt', 'w')
for i in range(res.__len__()):
    f.writelines([str(arr[i]) + ',' + str(res[i]) + '\n'])
