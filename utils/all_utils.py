import pandas as pd
import matplotlib.pyplot as plt
import os
from utils import config

def save_model(model):
  model_dir = 'models'
  os.makedirs(model_dir, exist_ok=True) # ONLY CREATE IF MODEL_DIR DOESN"T EXISTS
  model.save(config.TRAINED_MODEL_DIR)

def performance_plot(history):
    pd.DataFrame(history.history).plot(figsize = (10,7))
    plt.grid(True)
    plot_dir = "plots"
    os.makedirs(plot_dir, exist_ok=True) # ONLY CREATE IF MODEL_DIR DOESN"T EXISTS
    plt.savefig(config.PLOT_DIR)

def save_history(history):
    hist_dir = "history"
    os.makedirs(hist_dir, exist_ok=True) # ONLY CREATE IF MODEL_DIR DOESN"T EXISTS
    with open('history/history.txt', 'w') as f:
        f.write(str(pd.DataFrame(history)))

def logs():
    None