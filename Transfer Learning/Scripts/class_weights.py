import pandas as pd



df = pd.read_csv(

    "/data/leuven/374/vsc37499/thesis/tables/train_windows.csv"

)



neg = (df["label"] == 0).sum()

pos = (df["label"] == 1).sum()



print("NEG:", neg)

print("POS:", pos)



weight = neg / pos



print("POSITIVE CLASS WEIGHT:")

print(weight)
