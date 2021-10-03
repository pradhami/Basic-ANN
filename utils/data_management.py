import tensorflow as tf
import logging

def train_valid_test_generator():
    """Loading mnist dataset from tensorflow.keras and scaling the pixels between 0 to 1
    Args:
        NA
    Returns:
        nd array: Train, Test and Validation datasets
    """
    mnist = tf.keras.datasets.mnist
    (x_train_full, y_train_full), (x_test,y_test)= mnist.load_data()
    x_valid, x_train = x_train_full[:5000] /255, x_train_full[5000:] / 255
    y_valid, y_train = y_train_full[:5000], y_train_full[5000:]
    x_test = x_test/255

    return ((x_train,y_train),(x_valid,y_valid),(x_test,y_test))

