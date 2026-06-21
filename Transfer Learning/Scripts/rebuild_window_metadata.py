import os

import glob

import pandas as pd



WINDOW_DIR = (

    "/data/leuven/374/vsc37499/thesis/windows"

)



rows = []



files = glob.glob(

    os.path.join(

        WINDOW_DIR,

        "*.npy"

    )

)



print("FILES FOUND:", len(files))



for f in files:



    name = os.path.basename(f)



    patient, window = name.rsplit("_w", 1)



    window = int(

        window.replace(".npy", "")

    )



    rows.append(

        [patient, window]

    )



df = pd.DataFrame(

    rows,

    columns=[

        "patient",

        "window"

    ]

)



df = df.sort_values(

    ["patient", "window"]

)



out = (

"/data/leuven/374/vsc37499/"

"thesis/tables/window_metadata_v2.csv"

)



df.to_csv(

    out,

    index=False

)



print()

print("ROWS:", len(df))

print("PATIENTS:", df["patient"].nunique())

print()

print("SAVED:", out)
