import csv
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


class Dataset():
    def __init__(self, root):
        self.root = root
        self.dateline = self.read_data(self.root, 14, 1)
        self.dataset = self.build_dataset()
        self.length = len(self.dataset)
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

    def str2float(self, value):
        matches = re.findall(r'\d+\.\d+', value)
        res = float(matches[0])

        return res

    def read_params(self):
        with open(self.root) as csv_file:
            csv_reader = csv.reader(csv_file)
            rows = list(csv_reader)

        x_resolution = self.str2float(rows[3].__getitem__(0))
        y_resolution = self.str2float(rows[4].__getitem__(0))

        x_calibration = self.str2float(rows[8].__getitem__(0))
        y_calibration = self.str2float(rows[9].__getitem__(0))

        return x_resolution, y_resolution, x_calibration, y_calibration

    def read_data(self, fname, skiprows, nrows):
        lst = []

        with open(fname):
            data = pd.read_csv(fname, skiprows=skiprows, nrows=nrows)
            for elem in data:
                if self.is_float(elem):
                    lst.append(float(elem))
            print(f'Num of row in {fname}: {skiprows + 1}')
            print(lst)

        return lst

    def build_dataset(self):
        ds = []
        skiprows = 15
        data = self.read_data(self.root, skiprows, 2)
        for elem in data:
            if self.is_float(elem):
                ds.append(float(elem))
        print(f'Num of row in {self.root}: {skiprows + 1}')
        print(ds)

        x_resolution, y_resolution, x_calibration, y_calibration = self.read_params()
        scaled_positions = [position * x_resolution * x_calibration for position in ds]

        return scaled_positions


def minmax_normalize(self):
    for index in range(self.length):
        self.dataset[:, index] = (self.dataset[:, index] - self.dataset[:, index].min()) / (
                self.dataset[:, index].max() - self.dataset[:, index].min())


if __name__ == '__main__':
    # ./data
    dataset = Dataset(r'data/lines-pos.csv')
    # plt.plot(dataset.dataset[:, 0].T)
    plt.title('Signals')
    plt.xlabel('Dateline, um')
    plt.ylabel('Positions, um')
    xi = list(range(len(dataset.dateline)))
    plt.xlim(0, 640)
    plt.plot(dataset.dataset)
    plt.xticks(np.arange(min(dataset.dateline), max(dataset.dateline) + 1, 20))
    plt.xticks(rotation=90)
    plt.show()
    plt.close()
