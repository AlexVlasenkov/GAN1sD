import torch
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd


class Dataset():
    def __init__(self, root):
        self.root = root
        self.dataset = self.build_dataset()
        self.length = self.dataset.shape[1]
        # self.minmax_normalize()

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        step = self.dataset[:, idx]
        step = torch.unsqueeze(step, 0)
        # target = self.label[idx]
        target = 0  # only one class
        return step, target

    def is_float(self, value):
        if value is None:
            return False
        try:
            float(value)
            return True
        except:
            return False

    def build_dataset(self):
        dataset = []
        fname = 'data/lines-pos.csv'
        skiprows = 15
        nrows = 2
        # for _file in os.listdir(self.root):
        #   sample = np.loadtxt(os.path.join(self.root, _file)).T
        #   dataset.append(sample)

        for _file in os.listdir(self.root):
            # 15 и 2 для Pos = 1
            data = pd.read_csv(fname, skiprows=skiprows, nrows=nrows)
            for elem in data:
                if self.is_float(elem):
                    dataset.append(float(elem))
            print(f'Num of row in {fname}: {skiprows + 1}')
            print(dataset)

        dataset = np.vstack(dataset).T
        dataset = torch.from_numpy(dataset).float()

        return dataset

    def minmax_normalize(self):
        for index in range(self.length):
            self.dataset[:, index] = (self.dataset[:, index] - self.dataset[:, index].min()) / (
                    self.dataset[:, index].max() - self.dataset[:, index].min())


if __name__ == '__main__':
    # ./data
    dataset = Dataset(r'data')
    plt.plot(dataset.dataset[:, 0].T)
    plt.show()
