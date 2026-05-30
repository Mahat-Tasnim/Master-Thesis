import os

import numpy as np

import pandas as pd



import torch

from torch.utils.data import Dataset





WINDOW_DIR = (

    "/data/leuven/374/vsc37499/thesis/windows"

)





class EEGDataset(Dataset):



    def __init__(self, csv_file):



        self.df = pd.read_csv(csv_file)



    def __len__(self):



        return len(self.df)



    def __getitem__(self, idx):



        row = self.df.iloc[idx]



        patient = row["patient"]

        window  = row["window"]



        file_path = os.path.join(

            WINDOW_DIR,

            f"{patient}_w{window}.npy"

        )



        x = np.load(file_path)



        x = torch.tensor(

            x,

            dtype=torch.float32

        )


        y = torch.tensor(

            int(row["label"]),

            dtype=torch.long

        )


        return x, y
