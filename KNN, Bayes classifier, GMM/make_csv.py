import pandas as pd 
import numpy as np 
import glob


classes = ["coast", "insidecity", "mountain", "opencountry", "street"]
for label in classes:
    files = glob.glob("Dataset 2B/" + label + "/dev/*")
    df = [pd.read_csv(f, delimiter=' ',header=None).to_numpy().flatten() for f in files]

    val = pd.DataFrame(df)
    val.to_csv("Dataset 2B CSV/dev/" + label + ".csv")

