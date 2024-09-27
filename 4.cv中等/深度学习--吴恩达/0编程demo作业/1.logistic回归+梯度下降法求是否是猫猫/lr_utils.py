import numpy as np
import h5py
    
    
def load_dataset():
    train_dataset = h5py.File('D:/cv计算机视觉/cv-computer-visual/4.cv中等/深度学习--吴恩达/0编程demo作业/3.建设自己的深层神经网络/datasets/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

    test_dataset = h5py.File('D:/cv计算机视觉/cv-computer-visual/4.cv中等/深度学习--吴恩达/0编程demo作业/3.建设自己的深层神经网络/datasets/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes
    
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes