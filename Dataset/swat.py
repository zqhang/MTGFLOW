import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd
import numpy as np


def loader_SWat(root, batch_size, window_size, stride_size,train_split,label=False):
    data = pd.read_csv(root,sep = ';', low_memory=False)
    Timestamp = pd.to_datetime(data["Timestamp"])
    data["Timestamp"] = Timestamp
    data = data.set_index("Timestamp")
    labels = [ int(l!= 'Normal' ) for l in data["Normal/Attack"].values]
    for i in list(data): 
        data[i]=data[i].apply(lambda x: str(x).replace("," , "."))
    data = data.drop(["Normal/Attack"] , axis = 1)
    data = data.astype(float)
    n_sensor = len(data.columns)
    #%%
    feature = data.iloc[:,:51]
    scaler = StandardScaler()
    norm_feature = scaler.fit_transform(feature)

    norm_feature = pd.DataFrame(norm_feature, columns= data.columns, index = Timestamp)
    norm_feature = norm_feature.dropna(axis=1)
    train_df = norm_feature.iloc[:int(train_split*len(data))]
    train_label = labels[:int(train_split*len(data))]
    print('trainset size',train_df.shape, 'anomaly ration', sum(train_label)/len(train_label))
   
    val_df = norm_feature.iloc[int(0.6*len(data)):int(train_split*len(data))]
    val_label = labels[int(0.6*len(data)):int(train_split*len(data))]
    
    test_df = norm_feature.iloc[int(train_split*len(data)):]
    test_label = labels[int(train_split*len(data)):]

    print('testset size',test_df.shape, 'anomaly ration', sum(test_label)/len(test_label))
    if label:
        train_loader = DataLoader(SWat_dataset(train_df,train_label, window_size, stride_size), batch_size=batch_size, shuffle=False)
    else:
        train_loader = DataLoader(SWat_dataset(train_df,train_label, window_size, stride_size), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(SWat_dataset(val_df,val_label, window_size, stride_size), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(SWat_dataset(test_df,test_label, window_size, stride_size), batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader, n_sensor

def loader_SWat_OCC(root, batch_size, window_size, stride_size,train_split,label=False):
    data = pd.read_csv("Data/input/SWaT_Dataset_Normal_v1.csv",sep = ',', low_memory=False)
    Timestamp = pd.to_datetime(data["Timestamp"])
    data["Timestamp"] = Timestamp
    data = data.set_index("Timestamp")
    labels = [ int(l!= 'Normal' ) for l in data["Normal/Attack"].values]
    for i in list(data): 
        data[i]=data[i].apply(lambda x: str(x).replace("," , "."))
    data = data.drop(["Normal/Attack"] , axis = 1)
    data = data.astype(float)
    n_sensor = len(data.columns)
    #%%
    feature = data.iloc[:,:51]
    scaler = StandardScaler()

    norm_feature = scaler.fit_transform(feature)
    norm_feature = pd.DataFrame(norm_feature, columns= data.columns, index = Timestamp)
    norm_feature = norm_feature.dropna(axis=1)
    train_df = norm_feature.iloc[:]
    train_label = labels[:]
    print('trainset size',train_df.shape, 'anomaly ration', sum(train_label)/len(train_label))

    val_df = norm_feature.iloc[int(train_split*len(data)):]
    val_label = labels[int(train_split*len(data)):]
    
    data = pd.read_csv('Data/input/SWaT_Dataset_Attack_v0.csv',sep = ';', low_memory=False)
    Timestamp = pd.to_datetime(data["Timestamp"])
    data["Timestamp"] = Timestamp
    data = data.set_index("Timestamp")
    labels = [ int(l!= 'Normal' ) for l in data["Normal/Attack"].values]
    for i in list(data): 
        data[i]=data[i].apply(lambda x: str(x).replace("," , "."))
    data = data.drop(["Normal/Attack"] , axis = 1)
    data = data.astype(float)
    n_sensor = len(data.columns)
 
    feature = data.iloc[:,:51]
    scaler = StandardScaler()
    norm_feature = scaler.fit_transform(feature)
    norm_feature = pd.DataFrame(norm_feature, columns= data.columns, index = Timestamp)
    norm_feature = norm_feature.dropna(axis=1)
    

    test_df = norm_feature.iloc[int(0.8*len(data)):]
    test_label = labels[int(0.8*len(data)):]

    print('testset size',test_df.shape, 'anomaly ration', sum(test_label)/len(test_label))
    if label:
        train_loader = DataLoader(SWat_dataset(train_df,train_label, window_size, stride_size), batch_size=batch_size, shuffle=False)
    else:
        train_loader = DataLoader(SWat_dataset(train_df,train_label, window_size, stride_size), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(SWat_dataset(val_df,val_label, window_size, stride_size), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(SWat_dataset(test_df,test_label, window_size, stride_size), batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader, n_sensor


class SWat_dataset(Dataset):
    def __init__(self, df, label, window_size=60, stride_size=10) -> None:
        super(SWat_dataset, self).__init__()
        self.df = df
        self.window_size = window_size
        self.stride_size = stride_size

        self.data, self.idx, self.label = self.preprocess(df,label)
        self.columns = np.append(df.columns, ["Label"])
        self.timeindex = df.index[self.idx]
    def preprocess(self, df, label):

        start_idx = np.arange(0,len(df)-self.window_size,self.stride_size)
        end_idx = np.arange(self.window_size, len(df), self.stride_size)
        
        delat_time =  df.index[end_idx]-df.index[start_idx]

        idx_mask = delat_time==pd.Timedelta(self.window_size,unit='s')

        start_index = start_idx[idx_mask]
        
        label = [0 if sum(label[index:index+self.window_size]) == 0 else 1 for index in start_index ]
        return df.values, start_idx[idx_mask], np.array(label)

    def __len__(self):

        length = len(self.idx)

        return length

    def __getitem__(self, index):
        #  N X K X L X D 
        """
        """
        start = self.idx[index]
        end = start + self.window_size
        data = self.data[start:end].reshape([self.window_size,-1, 1])
        return torch.FloatTensor(data).transpose(0,1), self.label[index], index

