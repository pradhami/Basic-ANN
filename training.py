from utils.data_management import train_valid_test_generator
from utils.model import model
from utils import config
from utils import all_utils

def train():
    (x_train,y_train),(x_valid,y_valid),(x_test,y_test) = train_valid_test_generator()
    model_classifier = model()
    model_classifier.compile(optimizer=config.OPTIMIZER, 
                                loss=config.LOSS_FUNCTION,
                                metrics=config.METRICS
                                )
    history = model_classifier.fit(x_train, y_train, epochs=config.EPOCHS, validation_data=(x_valid,y_valid), verbose=config.VERBOSE)

    all_utils.save_model(model_classifier)
    all_utils.save_history(history.history)
    all_utils.performance_plot(history)

if __name__ == '__main__':
    train()