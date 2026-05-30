import pandas as pd

import numpy as np



WINDOW_SEC = 3



EVENTS = (

"/data/leuven/374/vsc37499/"

"thesis/tables/event_labels.csv"

)



WINDOWS = (

"/data/leuven/374/vsc37499/"

"thesis/tables/window_metadata_v2.csv"

)



events = pd.read_csv(EVENTS)



windows = pd.read_csv(WINDOWS)





def to_seconds(x):



    if pd.isna(x):

        return np.nan



    x = str(x)



    x = (

        x.replace('"', '')

         .replace("”", "")

         .replace("'", "")

         .replace("’", "")

    )



    if len(x) < 2:

        return np.nan



    sec = int(x[-2:])

    minute = int(x[:-2])



    return minute * 60 + sec





events["start_sec"] = events["start"].apply(to_seconds)

events["end_sec"]   = events["end"].apply(to_seconds)



events = events.dropna(

    subset=[

        "start_sec",

        "end_sec"

    ]

)



windows["label"] = 0



events["edf"] = events["edf"].astype(str)

windows["patient"] = windows["patient"].astype(str)



for patient in events["edf"].unique():



    patient_events = events[

        events["edf"] == patient

    ]



    idxs = windows.index[

        windows["patient"] == patient

    ]



    for idx in idxs:



        w = windows.loc[idx, "window"]



        win_start = w * WINDOW_SEC

        win_end = (w + 1) * WINDOW_SEC



        positive = False



        for _, ev in patient_events.iterrows():



            overlap = (

                win_start < ev["end_sec"]

                and

                win_end > ev["start_sec"]

            )



            if overlap:

                positive = True

                break



        windows.loc[idx, "label"] = int(positive)



print()

print("LABEL COUNTS")

print(

    windows["label"].value_counts()

)



out = (

"/data/leuven/374/vsc37499/"

"thesis/tables/window_labels_v2.csv"

)



windows.to_csv(

    out,

    index=False

)



print()

print("SAVED:", out)
