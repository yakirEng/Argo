import cv2
import os
import random
import numpy as np
from numpy.linalg import inv
import time


train_path = '/Data/train'
validation_path = '/data/val2017'
test_path = '/data/test2017'



def ImagePreProcessing(image_name, path):
    img = cv2.imread(path + '/%s' % image_name, 0)
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


# save .npy files
def savedata(path):
    lst = os.listdir(path + '/')
    os.makedirs(path + 'processed/')
    new_path = path + 'processed/'
    for i in lst:
        np.save(new_path + '%s' % i[0:12], ImagePreProcessing(i, path))


def pre_processing(source_paths, target_paths):
    start_time = time.time()
    for path_idx, src_path in enumerate(source_paths):
        images_names = os.listdir(src_path)
        print(f'start processing images from {src_path} to {target_paths[path_idx]}')
        print(f'time stamp: {time.time() - start_time:.2f} seconds')
        for image_idx, image_name in enumerate(images_names):
            datum = ImagePreProcessing(image_name, src_path)
            np.save(target_paths[path_idx] + '/' + f'{image_idx}.npy', datum)
            print(f'image {image_idx} processed')








# savedata(train_path)
# savedata(validation_path)
# savedata(test_path)

source_paths = [r'D:\Academic\Msc\Thesis\AgroCode\Projection\Data\train', r'D:\Academic\Msc\Thesis\AgroCode\Projection\Data\test']
target_paths = [r"D:\Academic\Msc\Thesis\AgroCode\Projection\Data\processed\train", r'D:\Academic\Msc\Thesis\AgroCode\Projection\Data\processed\test']
pre_processing(source_paths, target_paths)