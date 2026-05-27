import pandas as pd

import numpy as np



WINDOW_SEC=30



EVENTS=(

"/data/leuven/374/vsc37499/"

"thesis/tables/event_labels.csv"

)



WINDOWS=(

"/data/leuven/374/vsc37499/"

"thesis/tables/window_metadata.csv"

)



events=pd.read_csv(EVENTS)



windows=pd.read_csv(WINDOWS)





def to_seconds(v):



    if pd.isna(v):

        return np.nan



    s=str(v).strip()



    # numeric already

    try:

        return float(s)

    except:

        pass



    s=(

        s.replace('"',"")

        .replace("”","")

        .replace("′","'")

        .replace("’","'")

    )



    if ":" in s:



        m,sec=s.split(":")



        return (

            float(m)*60

            +

            float(sec)

        )



    if "'" in s:



        m,sec=s.split("'")



        return (

            float(m)*60

            +

            float(sec)

        )



    try:

        return float(s)



    except:



        print("BAD:",repr(v))



        return np.nan





print()

print("CONVERTING TIMES")



events["start_sec"]=(

events["start"]

.apply(to_seconds)

)



events["end_sec"]=(

events["end"]

.apply(to_seconds)

)



events=events.dropna(

subset=[

"start_sec",

"end_sec"

]

)



print(

"events:",

len(events)

)



windows["patient"]=(

windows["patient"]

.astype(str)

)



events["edf"]=(

events["edf"]

.astype(str)

)



windows["label"]=0





for idx,row in windows.iterrows():



    patient=row["patient"]



    window=row["window"]



    start=window*WINDOW_SEC



    end=(

        window+1

    )*WINDOW_SEC



    subset=events[

        events["edf"]

        ==

        patient

    ]



    positive=False



    for _,ev in subset.iterrows():



        overlap=(

            start

            <

            ev["end_sec"]

        ) and (

            end

            >

            ev["start_sec"]

        )



        if overlap:



            positive=True



            break



    windows.loc[

        idx,

        "label"

    ]=int(positive)





print()

print(

windows["label"]

.value_counts()

)



out=(

"/data/leuven/374/vsc37499/"

"thesis/tables/"

"window_labels.csv"

)



windows.to_csv(

out,

index=False

)



print()

print(

"SAVED:",

out

)
