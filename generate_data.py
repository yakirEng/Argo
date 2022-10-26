import numpy as np
import cv2
import random
import os
import random
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

def ImagePreProcessing(image_name, path):
    img = cv2.imread(path + '\\%s' % image_name, 0)
    img = cv2.resize(img, (320, 240))

    rho = 32
    patch_size = 128
    top_point = (32, 32)
    left_point = (patch_size + 32, 32)
    bottom_point = (patch_size + 32, patch_size + 32)
    right_point = (32, patch_size + 32)
    test_image = img.copy()
    four_points = [top_point, left_point, bottom_point, right_point]

    perturbed_four_points = []
    for point in four_points:
        perturbed_four_points.append((point[0] + random.randint(-rho, rho), point[1] + random.randint(-rho, rho)))

    H = cv2.getPerspectiveTransform(np.float32(four_points), np.float32(perturbed_four_points))
    H_inverse = inv(H)

    warped_image = cv2.warpPerspective(img, H_inverse, (320, 240))

    # annotated_warp_image = warped_image.copy()

    Ip1 = test_image[top_point[1]:bottom_point[1], top_point[0]:bottom_point[0]]
    Ip2 = warped_image[top_point[1]:bottom_point[1], top_point[0]:bottom_point[0]]

    training_image = np.dstack((Ip1, Ip2))
    H_four_points = np.subtract(np.array(perturbed_four_points), np.array(four_points))
    datum = (training_image, H_four_points)
    return datum


def pre_processing(source_paths, target_paths):
    start_time = time.time()
    for dir_idx, im_names in enumerate(images_names):
      print(f'there is {len(im_names)} images in this directory')
      print(f'loading from {source_paths[dir_idx]} to {target_paths[dir_idx]}' )
      for im_idx, im_name in enumerate(im_names):
        datum = ImagePreProcessing(im_name, source_paths[dir_idx])
        np.save(target_paths[dir_idx] + '/' + f'{im_idx}.npy', datum)
        print(f'image {im_idx} processed')
        if im_idx == len(im_names):
          print('!!!       FINISHED DIRECTORY      !!!')
          break