import os
from argparse import Namespace

import numpy as np
from PIL import Image
from tqdm import tqdm

from src.cli.defaults import DATA_ROOT


def pull_keras_dataset(dataset_name: str) -> None:
    """
    Pull a dataset from Keras
    :param dataset_name: The name of the dataset in Keras
    """
    import tensorflow.python.keras as keras
    dataset = getattr(keras.datasets, dataset_name)
    save_keras_local(dataset.load_data(), dataset_name)


def save_keras_local(data: tuple, dataset_name: str) -> None:
    """
    Saves a dataset from Keras in the local data directory
    :param data: The tuple of (train_x, train_y), (test_x, test_y) data
    :param dataset_name: The name of the dataset in Keras
    """
    full_x, full_y = combine_train_and_test(data)
    for i, x in enumerate(tqdm(full_x, unit="images", leave=False)):
        label = str(int(full_y[i]))
        dname = os.path.join(DATA_ROOT, dataset_name, label)
        if not os.path.exists(dname):
            os.makedirs(dname)
        fname = os.path.join(dname, "{:05d}.jpg".format(i))

        if not os.path.exists(fname):
            Image.fromarray(x).save(fname)


def combine_train_and_test(data: tuple) -> tuple:
    """
    Combines the test and train elements of a dataset to one x and y
    :param data: Tuple of (train_x, train_y), (test_x, test_y) data
    :return: (x, y) tuple
    """
    (train_x, train_y), (test_x, test_y) = data
    full_x = np.vstack((train_x, test_x))

    try:
        full_y = np.hstack((train_y, test_y))
    except ValueError:
        full_y = np.vstack((train_y, test_y))

    return full_x, full_y


def run(args: Namespace) -> None:
    if args.download:
        pull_keras_dataset(args.download)
