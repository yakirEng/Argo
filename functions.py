# import cv2
import os
import random
from datetime import datetime
import numpy as np
import time
from numpy.linalg import inv
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torchinfo import summary
import torchvision
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import json
from alive_progress import alive_bar

from homography_classes import CocoDataset, RegressionModel


def visualize(path):
    data = np.load(path, allow_pickle=True)
    orig = data[0][:, :, 0]
    homographed = data[0][:, :, 1]
    fig = plt.figure(figsize=(10, 7))
    fig.add_subplot(1, 2, 1)
    plt.imshow(orig, cmap='gray')
    plt.axis('off')
    plt.title('original')
    fig.add_subplot(1, 2, 2)
    plt.imshow(homographed, cmap='gray')
    plt.axis('off')
    plt.title('homographed')
    plt.show()
    print('and the homography points: \n' + str(data[1]))


def plot_loss(train_losses, eval_losses, plot_path, index):
    plt.plot(train_losses, label='train')
    plt.plot(eval_losses, label='validation')
    plt.grid()
    plt.title('MSE loss')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend()
    plt.savefig(os.path.join(plot_path,'plot.png'))
    plt.clf()


def train(TrainingData, ValidationData, device, plot_path):
    batch_size = 64
    TrainLoader = DataLoader(TrainingData, batch_size)
    ValidationLoader = DataLoader(ValidationData, batch_size)
    criterion = nn.MSELoss()
    num_samples = 118287
    total_iteration = 90000
    steps_per_epoch = num_samples / batch_size
    epochs = int(total_iteration / steps_per_epoch)
    model = RegressionModel().to(device)
    summary(model, input_size=(batch_size, 2, 128, 128))
    optimizer = optim.SGD(model.parameters(), lr=0.005, momentum=0.9)
    # scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=8, gamma=0.1)
    train_loss_vec, val_loss_vec, train_losses, val_losses = [], [], [], []
    min_loss = 9999
    for epoch in range(epochs):
        with alive_bar(len(TrainLoader), title=f'train {epoch + 1}:') as bar:
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
                # scheduler.step()
                train_loss_vec.append(loss.cpu().detach().numpy())
                bar()
        model.eval()
        with alive_bar(len(ValidationLoader), title=f'valid {epoch + 1}:', ) as bar:
            with torch.no_grad():
                for i, (images, target) in enumerate(ValidationLoader):
                    images = images.to(device)
                    target = target.to(device)
                    images = images.permute(0, 3, 1, 2).float()
                    target = target.float()
                    outputs = model(images)
                    loss = criterion(outputs, target.view(-1, 8))
                    val_loss_vec.append(loss.cpu().detach().numpy())
                    bar()
            train_loss = np.mean(train_loss_vec)
            val_loss = np.mean(val_loss_vec)


        print('Train Epoch: [{}/{}] [{}/{} ({:.0f}%)], train loss: {:.6f}, val loss: {:.6f} (MSE)'.format(
            epoch + 1, epochs, i, len(TrainLoader),
            100. * i / len(TrainLoader), train_loss, val_loss))

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if epoch % 1 == 0:
            plot_loss(train_losses, val_losses, plot_path, index=epoch)

        if val_loss < min_loss:
            state = {'epoch': epochs, 'state_dict': model.state_dict(),
                     'optimizer': optimizer.state_dict()}
            torch.save(state, os.path.join(plot_path, 'DeepHomographyEstimation.pth'))
            min_loss = val_loss

        train_loss_vec = []
        val_loss_vec = []


def test(TestData, device):
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
            tst_loss_vec.append(loss.cpu().numpy())
        print('test loss: ' + np.mean(tst_loss_vec))


def create_plot_path():
    path = r'E:\projection_task\report\plots'
    date = datetime.today().strftime('%Y-%m-%d')
    directory = os.path.join(path, date)
    if not os.path.exists(directory):
        os.makedirs(directory)
    running_directory = os.path.join(directory, str(len(os.listdir(directory)) + 1))
    os.makedirs(running_directory)
    return running_directory


def get_paths():
    data_path = 'E:\projection_task\data'
    processed_path = 'E:\projection_task\data\processed'

    train_path = os.path.join(data_path, 'train2017')
    validation_path = os.path.join(data_path, 'val2017')
    test_path = os.path.join(data_path, 'test2017')

    processed_train_path = os.path.join(processed_path, 'train')
    processed_validation_path = os.path.join(processed_path, 'val')
    processed_test_path = os.path.join(data_path, 'test')

    source_paths = [train_path, validation_path, test_path]
    target_paths = [processed_train_path, processed_validation_path, processed_test_path]

    plot_path = create_plot_path()
    return source_paths, target_paths, plot_path

def start_gpu():
    if torch.cuda.is_available():
        print('GPU is Available')
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_data(path):
    data = CocoDataset(path=path)
    return data