import numpy as np
import h5py


def load_data():

    train_dataset = h5py.File('data/train_dog.h5', "r")
    train_tensors = np.array(train_dataset["train_set_x"][:])  # train set features
    train_targets = np.array(train_dataset["train_set_y"][:])  # train set labels

    valid_dataset = h5py.File('data/valid_dog.h5', "r")
    valid_tensors = np.array(valid_dataset["valid_set_x"][:])  # validation set features
    valid_targets = np.array(valid_dataset["valid_set_y"][:])  # validation set labels

    test_dataset = h5py.File('data/test_dog.h5', "r")
    test_tensors = np.array(test_dataset["test_set_x"][:])  # test set features
    test_targets = np.array(test_dataset["test_set_y"][:])  # test set labels

    return train_tensors, train_targets, valid_tensors, valid_targets, test_tensors, test_targets

