import pickle

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

if __name__ =="__main__":

    modelToCheck = Path("./scores")
    modelToCheck = modelToCheck /   \
                   "FCNs-BCEWithLogits_batch2_epoch150_RMSprop_scheduler-step50-gamma0.5_lr0.0001_momentum0_w_decay1e-05"

    meanPixel = np.load(modelToCheck / "meanPixel.npy")
    lossLog = pickle.load(open(modelToCheck / "loss_log.pickle","rb"))


    fig, ax = plt.subplots(1,2)
    ax[0].set_title("Mean pixel")
    ax[0].plot(meanPixel)
    ax[1].set_title("Loss log")
    ax[1].plot(lossLog)
    plt.show()
    x = 0