import tensorflow as tf
from utils import config

def model():
    LAYERS = [
            tf.keras.layers.Flatten(input_shape=config.INPUT_SHAPE, name = 'InputLayer'),
            tf.keras.layers.Dense(300, activation='relu', name = 'HiddenLayer1'),
            tf.keras.layers.Dense(100, activation='relu', name = 'HiddenLayer2'),
            tf.keras.layers.Dense(config.CLASSES, activation='softmax', name = 'OutputLayer')
            ]

    model_classifier = tf.keras.models.Sequential(LAYERS)
    
    return(model_classifier)

    
