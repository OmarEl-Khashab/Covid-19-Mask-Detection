import os
import numpy as np
import shutil


def save_split(images, split_name, state, input_path, output_path):
    save_path = os.path.join(output_path, split_name, state)
    os.makedirs(save_path)
    for im in images:
        shutil.copyfile(os.path.join(input_path, im), os.path.join(save_path, im))


# data_dir = "D:/Mask Detection"
data_dir = "/home/abdalla/Omar/mask_detection"
states = ["with_mask", "without_mask"]

for s in states:
    images_path = os.path.join(data_dir, s)
    num_files = len(images_path)
    images = os.listdir(images_path)
    np.random.shuffle(images)
    train_size = int(0.8 * len(images))
    valid_size = int(0.1 * len(images))
    test_size = int(0.1 * len(images))

    train_images = images[:train_size]

    valid_images = images[train_size:train_size + valid_size]

    test_images = images[train_size + valid_size:]

    save_split(train_images, "train", s, images_path, data_dir)
    save_split(valid_images, "validation", s, images_path, data_dir)
    save_split(test_images, "test", s, images_path, data_dir)
