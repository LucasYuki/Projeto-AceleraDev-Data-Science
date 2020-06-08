# -*- coding: utf-8 -*-
"""
Created on Fri May 29 16:29:38 2020

@author: lucas
"""

import pandas as pd
from matplotlib import pyplot as plt

def plot_train_hist(path, figsize=(10,5)):
    data = pd.read_csv(path+"\\train_history.csv")

    fig, ax = plt.subplots(figsize=figsize)  # Create a figure and an axes.
    ax.plot(data["epoch"], data["loss"], label='train', lw=0.5)  # Plot some data on the axes.
    ax.plot(data["epoch"], data["val_loss"], label='validate', lw=0.5)  # Plot more data on the axes...
    ax.set_xlabel('epoch')  # Add an x-label to the axes.
    ax.set_ylabel('loss')  # Add a y-label to the axes.
    ax.set_title("loss")  # Add a title to the axes.
    ax.legend()  # Add a legend.
    plt.grid(True)
    plt.savefig(path+"//train_history_loss.png", quality=100)
    plt.savefig(path+"//train_history_loss.SVG", quality=100)
    plt.show()
    plt.close()
    