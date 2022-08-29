import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np
import os
import torch
import torch.optim as optim
from torchsummary import summary
import torchvision
from torch.utils.data import Dataset, DataLoader

from homography_models import RegressionModel


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CocoDataset():
    def __init__(self, path):
        X = ()
        Y = ()
        lst = os.listdir(path)
        it = 0
        for i in lst:
            array = np.load(path + '%s' % i)
            x = torch.from_numpy((array[0].astype(float) - 127.5) / 127.5)
            X = X + (x,)
            y = torch.from_numpy(array[1].astype(float) / 32.)
            Y = Y + (y,)
            it += 1
        self.len = it
        self.X_data = X
        self.Y_data = Y

    def __getitem__(self, index):
        return self.X_data[index], self.Y_data[index]

    def __len__(self):
        return self.len


train_path = '/home/jupyter/train2017/train2017processed/'
validation_path = '/home/jupyter/val2017/val2017processed/'
test_path = '/home/jupyter/test2017/test2017processed/'

TrainingData = CocoDataset(train_path)
ValidationData = CocoDataset(validation_path)
TestData = CocoDataset(test_path)

# Training
def train():
    batch_size = 64
    TrainLoader = DataLoader(TrainingData, batch_size)
    ValidationLoader = DataLoader(ValidationData, batch_size)
    criterion = nn.MSELoss()
    num_samples = 118287
    total_iteration = 90000
    steps_per_epoch = num_samples / batch_size
    epochs = int(total_iteration / steps_per_epoch)
    model = RegressionModel().to(device)
    summary(model, (2, 128, 128))
    optimizer = optim.SGD(model.parameters(), lr=0.005, momentum=0.9)
    train_loss, val_loss, train_losses, val_losses = [], [], [], []
    min_loss = 9999
    for epoch in range(epochs):
        for i, (images, target) in enumerate(TrainLoader):
            optimizer.zero_grad()
            images = images.to(device);
            target = target.to(device)
            images = images.permute(0, 3, 1, 2).float();
            target = target.float()
            outputs = model(images)
            loss = criterion(outputs, target.view(-1, 8))
            loss.backward()
            optimizer.step()
            train_loss_vec.append(loss)

        model.eval()
        with torch.no_grad():
            for i, (images, target) in enumerate(ValidationLoader):
                images = images.to(device)
                target = target.to(device)
                images = images.permute(0, 3, 1, 2).float()
                target = target.float()
                outputs = model(images)
                loss = criterion(outputs, target.view(-1, 8))
                val_loss_vec.append(loss)

        train_loss = np.mean(train_loss_vec)
        val_loss = np.mean(val_loss_vec)

        print('Train Epoch: [{}/{}] [{}/{} ({:.0f}%)]\,MSE train loss: {:.6f}, MSE val loss: {:.6f}'.format(
            epoch + 1, epochs, i, len(TrainLoader),
            100. * i / len(TrainLoader), train_loss, val_loss))

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        plot_loss(epochs, train_losses, val_losses)

        if np.mean(val_loss) < min_loss:
            state = {'epoch': epochs, 'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict()}
            torch.save(state, 'DeepHomographyEstimation.pth')
            min_loss = np.mean(val_loss)

        train_loss_vec = []
        val_loss_vec = []

def test():
    batch_size = 64
    TestLoader = DataLoader(TestData, batch_size)
    model = RegressionModel().to(device)
    criterion = nn.MSELoss()
    model.eval()
    tst_loss_vec = []
    with torch.no_grad():
        for i, (images, target) in enumerate(TestLoader):
            images = images.to(device)
            target = target.to(device)
            images = images.permute(0, 3, 1, 2).float()
            target = target.float()
            outputs = model(images)
            loss = criterion(outputs, target.view(-1, 8))
            tst_loss_vec.append(loss)
        print('test loss: ' + np.mean(tst_loss_vec))



def plot_loss(epochs, train_loss, eval_loss):
    plt.plot(range(epochs), train_loss)
    plt.plot(range(epochs), eval_loss)
    plt.grid()
    plt.title('MSE loss')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend('train', 'validation')
