import numpy as np

import pandas as pd

import torch

from sklearn.metrics import (

    roc_auc_score,

    confusion_matrix,

    precision_score,

    f1_score

)



ROOT="/data/leuven/374/vsc37499/thesis"



MODEL=f"{ROOT}/models/cnn_real.pth"



WINDOWS=f"{ROOT}/windows"



SPLIT=f"{ROOT}/tables/dataset_split.csv"





class CNN(torch.nn.Module):



    def __init__(self):



        super().__init__()



        self.conv=torch.nn.Conv1d(

            16,

            32,

            7

        )



        self.pool=torch.nn.AdaptiveAvgPool1d(

            100

        )



        self.fc=torch.nn.Linear(

            3200,

            2

        )



    def forward(self,x):



        x=self.conv(x)



        x=torch.relu(x)



        x=self.pool(x)



        x=x.reshape(

            x.shape[0],

            -1

        )



        x=self.fc(x)



        return x





class EEGDataset(

    torch.utils.data.Dataset

):



    def __init__(self):



        df=pd.read_csv(

            SPLIT

        )



        self.df=df[

            df.split=="test"

        ].reset_index(

            drop=True

        )



    def __len__(self):



        return len(

            self.df

        )



    def __getitem__(self,idx):



        r=self.df.iloc[idx]



        x=np.load(

            f"{WINDOWS}/{r.patient}_w{r.window}.npy"

        )



        x=torch.tensor(

            x,

            dtype=torch.float32

        )



        y=int(

            r.label

        )



        return x,y





print(

"LOAD MODEL"

)



model=CNN()



model.load_state_dict(

    torch.load(

        MODEL,

        map_location="cpu"

    )

)



model.eval()



ds=EEGDataset()



loader=torch.utils.data.DataLoader(

    ds,

    batch_size=64

)



scores=[]



labels=[]



print(

"EVALUATING"

)



with torch.no_grad():



    for x,y in loader:



        out=model(

            x

        )



        p=torch.softmax(

            out,

            dim=1

        )[:,1]



        scores.extend(

            p.numpy()

        )



        labels.extend(

            y.numpy()

        )



scores=np.array(

scores

)



labels=np.array(

labels

)



pred=(

scores>=0.5

).astype(int)



tn,fp,fn,tp=(

confusion_matrix(

labels,

pred

).ravel()

)



print()



print(

"AUC",

round(

roc_auc_score(

labels,

scores

),

4

)

)



print(

"SENSITIVITY",

round(

tp/(tp+fn),

4

)

)



print(

"SPECIFICITY",

round(

tn/(tn+fp),

4

)

)



print(

"PPV",

round(

precision_score(

labels,

pred

),

4

)

)



print(

"F1",

round(

f1_score(

labels,

pred

),

4

)

)



print()



print(

[[tn,fp],

 [fn,tp]]

)
