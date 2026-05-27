import os

import numpy as np

import pandas as pd

import torch

import torch.nn as nn



from torch.utils.data import Dataset

from torch.utils.data import DataLoader





ROOT="/data/leuven/374/vsc37499/thesis"



META=f"{ROOT}/tables/dataset_split.csv"



WINDOW_DIR=f"{ROOT}/windows"



MODEL_OUT=f"{ROOT}/models/cnn_real.pth"



DEVICE="cpu"



BATCH=8



EPOCHS=15





class EEGDataset(Dataset):



    def __init__(self, split):



        df=pd.read_csv(META)



        self.df=(

            df[

                df["split"]==split

            ]

            .reset_index(drop=True)

        )



    def __len__(self):



        return len(self.df)



    def __getitem__(self, idx):



        row=self.df.iloc[idx]



        file=(

            f"{WINDOW_DIR}/"

            f"{row['patient']}"

            f"_w"

            f"{int(row['window'])}"

            f".npy"

        )



        x=np.load(file)



        if x.shape!=(16,15000):



            raise Exception(

                f"bad shape: {file} {x.shape}"

            )



        x=torch.tensor(

            x,

            dtype=torch.float32

        )



        y=torch.tensor(

            int(row["label"]),

            dtype=torch.long

        )



        return x,y





class CNN(nn.Module):



    def __init__(self):



        super().__init__()



        self.conv=nn.Conv1d(

            16,

            32,

            kernel_size=7

        )



        self.pool=nn.AdaptiveAvgPool1d(

            100

        )



        self.fc=nn.Linear(

            3200,

            2

        )



    def forward(self,x):



        x=self.conv(x)



        x=torch.relu(x)



        x=self.pool(x)



        x=x.flatten(1)



        x=self.fc(x)



        return x





print()

print("LOADING DATA")



train_loader=DataLoader(

    EEGDataset("train"),

    batch_size=BATCH,

    shuffle=True

)



val_loader=DataLoader(

    EEGDataset("val"),

    batch_size=BATCH

)



print("DATA READY")



model=CNN()



weights=torch.tensor(

    [1.0,45.0]

)



loss_fn=nn.CrossEntropyLoss(

    weight=weights

)



opt=torch.optim.Adam(

    model.parameters(),

    lr=0.001

)



best=999999





print()

print("TRAINING")



for epoch in range(EPOCHS):



    model.train()



    train_loss=0



    for x,y in train_loader:



        opt.zero_grad()



        pred=model(x)



        loss=loss_fn(

            pred,

            y

        )



        loss.backward()



        opt.step()



        train_loss+=loss.item()



    train_loss/=len(train_loader)



    model.eval()



    val_loss=0



    with torch.no_grad():



        for x,y in val_loader:



            pred=model(x)



            loss=loss_fn(

                pred,

                y

            )



            val_loss+=loss.item()



    val_loss/=len(val_loader)



    print(

        f"Epoch {epoch+1}"

        f" | train={train_loss:.4f}"

        f" | val={val_loss:.4f}"

    )



    if val_loss<best:



        best=val_loss



        torch.save(

            model.state_dict(),

            MODEL_OUT

        )





print()

print("MODEL SAVED")



print(MODEL_OUT)



print()



print("TRAINING COMPLETE")
