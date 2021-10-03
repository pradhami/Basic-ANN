import os

BATCH_SIZE = 32
EPOCHS = 10
CLASSES = 10
INPUT_SHAPE = [28,28]
LOSS_FUNCTION = "sparse_categorical_crossentropy"
OPTIMIZER = "SGD"
METRICS = ["accuracy"]
TRAINED_MODEL_DIR = os.path.join("models", "ANN.h5")
LOGGING_DIR = os.path.join("logs","running_logs.log")
PLOT_DIR = os.path.join('plots', 'performance.png')
VERBOSE = 0