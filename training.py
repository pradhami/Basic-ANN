from utils.data_management import train_valid_test_generator
from utils.model import model
from utils import config
from utils import all_utils
import logging
import os

logging_str = "[%(asctime)s: %(levelname)s: %(module)s] %(message)s"
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(filename = config.LOGGING_DIR,level=logging.INFO, format=logging_str)

def train():
    logging.info(f"----------Preparing Train, Test and Validation set----------")
    (x_train,y_train),(x_valid,y_valid),(x_test,y_test) = train_valid_test_generator()
    logging.info(f"----------Train, Test and Validation set created successfully----------")

    logging.info(f"----------Defining Model Parameters----------")
    model_classifier = model()
    model_classifier.compile(optimizer=config.OPTIMIZER, 
                                loss=config.LOSS_FUNCTION,
                                metrics=config.METRICS
                                )
    logging.info(f"----------Model Parameters Defination Completed----------")

    logging.info(f"----------Training Starting----------")
    history = model_classifier.fit(x_train, y_train, epochs=config.EPOCHS, validation_data=(x_valid,y_valid), verbose=config.VERBOSE)
    logging.info(f"----------Training Finished----------")

    logging.info(f"----------Saving Model----------")
    all_utils.save_model(model_classifier)
    logging.info(f"----------Saving Model Completed----------")

    logging.info(f"----------Saving History----------")
    all_utils.save_history(history.history)
    logging.info(f"----------Saving History Completed----------")

    logging.info(f"----------Saving Performance Plot----------")
    all_utils.performance_plot(history)
    logging.info(f"----------Saving Performance Plot Completed----------")

if __name__ == '__main__':
    train()