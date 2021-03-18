import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

if __name__ == "__main__":

    filesPath = Path(r"C:\Users\asus\Downloads\2021-03-08_Alfredo_collection_1\2021-03-08_19h.01m.51s_AlfredoManual_02")
    bloodTxt = filesPath / "blood_percentage.txt"

    df = pd.read_csv(bloodTxt, index_col=[0])
    plt.plot(df['blood_percentage'])
    plt.show()
    x =1

