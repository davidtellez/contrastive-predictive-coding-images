"""
Preprocesses and prepares the MNIST dataset for the experiments.
"""

from os.path import join, basename, dirname, exists
import numpy as np
import skimage.color
import os
import gzip
import scipy.ndimage
from PIL import Image
from tqdm import tqdm
import skimage.color
import scipy.ndimage
import skimage.transform
from scipy.misc import imsave


def create_mnist_dataset_npy(input_dir, output_dir):
    """
    Preprocesses MNIST digits and store them in a suitable format. It resamples the images to 64x64 and adds a colorful
    background to make the classification task more challenging. It requires <1GB of storage.

    :param input_dir: path to folder containing the MNIST source data files and lena.jpg.
    :param output_dir: path to destination folder where the following files will be created: training_x.npy,
    training_y.npy, validation_x.npy, validation_y.npy, test_x.npy, and test_y.npy.
    :return: nothing.
    """

    def load_mnist_images(filename):

        # Read the inputs in Yann LeCun's binary format.
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        data = data.reshape(-1, 28, 28)
        return data / np.float32(256)

    def load_mnist_labels(filename):

        # Read the labels in Yann LeCun's binary format.
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=8)
        return data

    def preprocess_digit(patch, lena):

        # Resize to 64x64
        patch = scipy.ndimage.zoom(patch[:, :], 2.3, order=1)

        # Make RGB
        patch = np.stack([patch]*3, axis=-1)

        # Binarize
        patch[patch >= 0.5] = 1
        patch[patch < 0.5] = 0

        # Crop from lena
        x_c = np.random.randint(0, lena.size[0] - 64)
        y_c = np.random.randint(0, lena.size[1] - 64)
        image = lena.crop((x_c, y_c, x_c + 64, y_c + 64))
        image = np.asarray(image) / 255.0

        # Randomly alter the color distribution of the crop
        for j in range(3):
            image[:, :, j] = (image[:, :, j] + np.random.uniform(0, 1)) / 2.0

        # Invert the color of pixels where there is a number
        image[patch[:, :, :] == 1] = 1 - image[patch[:, :, :] == 1]
        patch[:, :, :] = image

        patch = np.floor(patch * 255).astype('uint8')

        return patch

    def save_array(x, y, output_x_path, output_y_path):

        idx = np.random.choice(x.shape[0], x.shape[0], replace=False)
        x = x[idx, ...]
        y = y[idx, ...]

        np.save(output_x_path, x.astype('uint8'))
        np.save(output_y_path, y.astype('uint8'))

    def save_images_to_disk(x, y, n, output_images_dir):

        if not exists(output_images_dir):
            os.makedirs(output_images_dir)

        idx = np.random.choice(len(x), n, replace=False)
        for i, (image, label) in enumerate(zip(x[idx], y[idx])):
            imsave(join(output_images_dir, '{i}_{label}.png'.format(i=i, label=label)), image)

    # Output dir
    if not exists(output_dir):
        os.makedirs(output_dir)

    # Read data
    x_train = load_mnist_images(join(input_dir, 'train-images-idx3-ubyte.gz'))
    y_train = load_mnist_labels(join(input_dir, 'train-labels-idx1-ubyte.gz'))
    x_test = load_mnist_images(join(input_dir, 't10k-images-idx3-ubyte.gz'))
    y_test = load_mnist_labels(join(input_dir, 't10k-labels-idx1-ubyte.gz'))
    x_train, x_val = x_train[:-10000], x_train[-10000:]
    y_train, y_val = y_train[:-10000], y_train[-10000:]

    # Preprocess patches
    lena = Image.open(join(input_dir, 'lena.jpg'))
    x_train = np.stack([preprocess_digit(patch, lena) for patch in tqdm(x_train)], axis=0)
    x_val = np.stack([preprocess_digit(patch, lena) for patch in tqdm(x_val)], axis=0)
    x_test = np.stack([preprocess_digit(patch, lena) for patch in tqdm(x_test)], axis=0)

    # Shuffle and store
    save_array(x_train, y_train, join(output_dir, 'training_x.npy'), join(output_dir, 'training_y.npy'))
    save_array(x_val, y_val, join(output_dir, 'validation_x.npy'), join(output_dir, 'validation_y.npy'))
    save_array(x_test, y_test, join(output_dir, 'test_x.npy'), join(output_dir, 'test_y.npy'))

    # Store images
    save_images_to_disk(x_train, y_train, 16, join(output_dir, 'training_images'))
    save_images_to_disk(x_val, y_val, 16, join(output_dir, 'validation_images'))
    save_images_to_disk(x_test, y_test, 16, join(output_dir, 'test_images'))


def augment_crops_mnist(crops):
    """ Performs cropping and HSV/HUE color augmentation in image crops. """

    def change_hsv(patch, h, s, v):

        # Convert
        patch_hsv = skimage.color.rgb2hsv(rgb=patch)

        # H Channel
        patch_hsv[:, :, 0] += h % 1.0
        patch_hsv[:, :, 0] %= 1.0

        # S Channel
        patch_hsv[:, :, 1] = np.clip(patch_hsv[:, :, 1] + s, 0, 1)

        # V Channel
        patch_hsv[:, :, 2] = np.clip(patch_hsv[:, :, 2] + v, 0, 1)

        # Convert back
        patch = skimage.color.hsv2rgb(hsv=patch_hsv)
        return patch

    # Pad
    crops_shape = crops.shape
    n = crops_shape[-3]
    p = crops_shape[2] // 8  # paper is //16
    crops = crops.reshape((-1,) + crops_shape[-3:])
    crops_pad = np.pad(crops, ((0, 0), (p, p), (p, p), (0, 0)), 'reflect')

    # Crop
    for i in range(crops_pad.shape[0]):
        crop = np.copy(crops_pad[i, ...])
        x = np.random.randint(0, p * 2 + 1)
        y = np.random.randint(0, p * 2 + 1)
        crop = crop[x:x + n, y:y + n, :]

        # Change color hue
        h = np.random.uniform(-1, 1)
        s = np.random.uniform(-0.5, 0.5)
        v = np.random.uniform(-0.5, 0.5)
        crop = change_hsv(crop * 0.5 + 0.5, h, s, v) * 2 - 1

        crops[i, ...] = crop

    # Reshape
    crops = crops.reshape(crops_shape)

    return crops


def augment_images_mnist(images):
    """ Performs simple image augmentation (cropping). """

    # Pad
    x_shape = images.shape
    n = x_shape[1]
    p = n // 6
    images_pad = np.pad(images, ((0, 0), (p, p), (p, p), (0, 0)), 'reflect')

    # Crop
    for i in range(images.shape[0]):
        x_i = np.random.randint(0, p * 2 + 1)
        y_i = np.random.randint(0, p * 2 + 1)
        im = images_pad[i, x_i:x_i + n, y_i:y_i + n, :]
        images[i, :, :, :] = im

    return images


if __name__ == '__main__':

    create_mnist_dataset_npy(
        input_dir=join('.', 'resources'),
        output_dir=join('.', 'resources', 'data')
    )
