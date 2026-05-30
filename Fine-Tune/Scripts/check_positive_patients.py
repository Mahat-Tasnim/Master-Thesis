import pandas as pd



df = pd.read_csv(

"/data/leuven/374/vsc37499/thesis/tables/window_labels_v2.csv"

)



print()

print("TOTAL POSITIVE WINDOWS")



print(

(df["label"]==1).sum()

)



print()

print("PATIENTS WITH POSITIVE WINDOWS")



print(

df[df["label"]==1]

["patient"]

.value_counts()

.head(30)

)
