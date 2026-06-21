import os

import numpy as np

import pandas as pd



import torch

import torch.nn as nn



from torch.utils.data import DataLoader



from sklearn.metrics import (

    roc_auc_score,

    f1_score,

    precision_score,

    confusion_matrix

)



from eeg_dataset import EEGDataset





# -----------------------

# DEVICE

# -----------------------



device = torch.device("cpu")





# -----------------------

# DATA

# -----------------------



train_dataset = EEGDataset(

    "/data/leuven/374/vsc37499/thesis/tables/train_windows.csv"

)



val_dataset = EEGDataset(

    "/data/leuven/374/vsc37499/thesis/tables/val_windows.csv"

)



train_loader = DataLoader(

    train_dataset,

    batch_size=64,

    shuffle=True

)



val_loader = DataLoader(

    val_dataset,

    batch_size=64,

    shuffle=False

)





# -----------------------

# MODEL

# -----------------------



class EEGCNN(nn.Module):



    def __init__(self):



        super().__init__()



        self.conv1 = nn.Conv1d(

            16,

            32,

            kernel_size=7

        )



        self.pool = nn.AdaptiveAvgPool1d(

            100

        )



        self.fc = nn.Linear(

            3200,

            2

        )



    def forward(self,x):



        x = torch.relu(

            self.conv1(x)

        )



        x = self.pool(x)



        x = x.flatten(1)



        x = self.fc(x)



        return x





model = EEGCNN().to(device)





# -----------------------

# LOSS

# -----------------------



weights = torch.tensor(

    [1.0,94.55],

    dtype=torch.float32

)



criterion = nn.CrossEntropyLoss(

    weight=weights

)



optimizer = torch.optim.Adam(

    model.parameters(),

    lr=1e-3

)





# -----------------------

# TRAINING

# -----------------------



best_auc = 0

patience = 5

counter = 0



os.makedirs(

    "/data/leuven/374/vsc37499/thesis/models",

    exist_ok=True

)



for epoch in range(20):



    model.train()



    train_loss = 0



    for x,y in train_loader:



        x = x.to(device)

        y = y.to(device)



        optimizer.zero_grad()



        logits = model(x)



        loss = criterion(

            logits,

            y

        )



        loss.backward()



        optimizer.step()



        train_loss += loss.item()



    train_loss /= len(train_loader)



    # -------------------

    # VALIDATION

    # -------------------



    model.eval()



    val_loss = 0



    probs_all = []

    preds_all = []

    labels_all = []



    with torch.no_grad():



        for x,y in val_loader:



            x = x.to(device)

            y = y.to(device)



            logits = model(x)



            loss = criterion(

                logits,

                y

            )



            val_loss += loss.item()



            probs = torch.softmax(

                logits,

                dim=1

            )[:,1]



            preds = torch.argmax(

                logits,

                dim=1

            )



            probs_all.extend(

                probs.cpu().numpy()

            )



            preds_all.extend(

                preds.cpu().numpy()

            )



            labels_all.extend(

                y.cpu().numpy()

            )



    val_loss /= len(val_loader)



    auc = roc_auc_score(

        labels_all,

        probs_all

    )



    f1 = f1_score(

        labels_all,

        preds_all,

        zero_division=0

    )



    ppv = precision_score(

        labels_all,

        preds_all,

        zero_division=0

    )



    tn, fp, fn, tp = confusion_matrix(

        labels_all,

        preds_all

    ).ravel()



    sensitivity = tp / (tp + fn + 1e-8)



    specificity = tn / (tn + fp + 1e-8)



    print(

        f"\nEpoch {epoch+1}"

    )



    print(

        f"Train Loss: {train_loss:.4f}"

    )



    print(

        f"Val Loss: {val_loss:.4f}"

    )



    print(

        f"AUC: {auc:.4f}"

    )



    print(

        f"Sensitivity: {sensitivity:.4f}"

    )



    print(

        f"Specificity: {specificity:.4f}"

    )



    print(

        f"PPV: {ppv:.4f}"

    )



    print(

        f"F1: {f1:.4f}"

    )



    if auc > best_auc:



        best_auc = auc



        counter = 0



        torch.save(

            model.state_dict(),

            "/data/leuven/374/vsc37499/thesis/models/baseline_cnn.pth"

        )



        print(

            "MODEL SAVED"

        )



    else:



        counter += 1



        if counter >= patience:



            print(

                "EARLY STOPPING"

            )



            break



print(

    "\nBEST AUC:",

    best_auc

)
