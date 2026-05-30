import pandas as pd

import numpy as np



np.random.seed(42)



df = pd.read_csv(

    "/data/leuven/374/vsc37499/thesis/tables/window_labels_v2.csv"

)



patients = sorted(df["patient"].unique())



np.random.shuffle(patients)



n = len(patients)



train_end = int(0.70*n)

val_end   = int(0.85*n)



train_patients = patients[:train_end]

val_patients   = patients[train_end:val_end]

test_patients  = patients[val_end:]



print("TRAIN:", len(train_patients))

print("VAL:", len(val_patients))

print("TEST:", len(test_patients))



pd.DataFrame(

    {"patient":train_patients}

).to_csv(

    "/data/leuven/374/vsc37499/thesis/tables/train_patients.csv",

    index=False

)



pd.DataFrame(

    {"patient":val_patients}

).to_csv(

    "/data/leuven/374/vsc37499/thesis/tables/val_patients.csv",

    index=False

)



pd.DataFrame(

    {"patient":test_patients}

).to_csv(

    "/data/leuven/374/vsc37499/thesis/tables/test_patients.csv",

    index=False

)
