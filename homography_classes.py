import torch
import torch.nn as nn
import os
import numpy as np
from alive_progress import alive_bar


class RegressionModel(nn.Module):
    def __init__(self):
        super(RegressionModel, self).__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(2, 64, 3, padding=1),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU())

        self.layer2 = nn.Sequential(nn.Conv2d(64, 64, 3, padding=1),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU(),
                                    nn.MaxPool2d(2),
                                    nn.Dropout(p=0.5))
        self.layer3 = nn.Sequential(nn.Conv2d(64, 64, 3, padding=1),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU())
        self.layer4 = nn.Sequential(nn.Conv2d(64, 64, 3, padding=1),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU(),
                                    nn.MaxPool2d(2),
                                    nn.Dropout(p=0.5))
        self.layer5 = nn.Sequential(nn.Conv2d(64, 128, 3, padding=1),
                                    nn.BatchNorm2d(128),
                                    nn.ReLU())
        self.layer6 = nn.Sequential(nn.Conv2d(128, 128, 3, padding=1),
                                    nn.BatchNorm2d(128),
                                    nn.ReLU(),
                                    nn.MaxPool2d(2),
                                    nn.Dropout(p=0.5))
        self.layer7 = nn.Sequential(nn.Conv2d(128, 128, 3, padding=1),
                                    nn.BatchNorm2d(128),
                                    nn.ReLU())
        self.layer8 = nn.Sequential(nn.Conv2d(128, 128, 3, padding=1),
                                    nn.BatchNorm2d(128),
                                    nn.ReLU(),
                                    nn.Dropout(p=0.5))
        self.fc1 = nn.Linear(128 * 16 * 16, 1024)
        self.fc2 = nn.Linear(1024, 8)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)
        out = out.reshape(-1, 128 * 16 * 16)
        out = self.fc1(out)
        out = self.fc2(out)
        return out


class CocoDataset():
    def __init__(self, path):
        # X = ()
        # Y = ()
        # lst = os.listdir(path)
        # it = 0
        # with alive_bar(len(lst), title='loadin data: ', force_tty=True) as bar:
        #     for i in lst:
        #         array = np.load(os.path.join(path, i), allow_pickle=True)
        #         x = torch.from_numpy((array[0].astype(float) - 127.5) / 127.5)
        #         X = X + (x,)
        #         y = torch.from_numpy(array[1].astype(float) / 32.)
        #         Y = Y + (y,)
        #         it += 1
        #         bar()

        # self.len = it
        # self.X_data = X
        # self.Y_data = Y
        self.path = path
        self.len = len(os.listdir(self.path))

    def __getitem__(self, index):
        array = np.load(os.path.join(self.path, str(index) + ".npy"), allow_pickle=True)
        x = torch.from_numpy((array[0].astype(float) - 127.5) / 127.5)
        y = torch.from_numpy(array[1].astype(float) / 32.)
        return x, y
        # return self.X_data, self.Y_data

    def __len__(self):
        return self.len