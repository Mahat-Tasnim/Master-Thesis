import pandas as pd



labels = pd.read_csv(

    "/data/leuven/374/vsc37499/thesis/tables/window_labels_v2.csv"

)



train_patients = set(

    pd.read_csv(

        "/data/leuven/374/vsc37499/thesis/tables/train_patients.csv"

    )["patient"]

)



val_patients = set(

    pd.read_csv(

        "/data/leuven/374/vsc37499/thesis/tables/val_patients.csv"

    )["patient"]

)



test_patients = set(

    pd.read_csv(

        "/data/leuven/374/vsc37499/thesis/tables/test_patients.csv"

    )["patient"]

)



train_df = labels[

    labels["patient"].isin(train_patients)

]



val_df = labels[

    labels["patient"].isin(val_patients)

]



test_df = labels[

    labels["patient"].isin(test_patients)

]



print("TRAIN:", len(train_df))

print("VAL:", len(val_df))

print("TEST:", len(test_df))



train_df.to_csv(

    "/data/leuven/374/vsc37499/thesis/tables/train_windows.csv",

    index=False

)



val_df.to_csv(

    "/data/leuven/374/vsc37499/thesis/tables/val_windows.csv",

    index=False

)



test_df.to_csv(

    "/data/leuven/374/vsc37499/thesis/tables/test_windows.csv",

    index=False

)
