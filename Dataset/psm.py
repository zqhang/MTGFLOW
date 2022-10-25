import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd
import numpy as np


def loader_PSM(root, batch_size, window_size, stride_size,train_split,label=False):
    data = pd.read_csv("Data/input/PSM/test.csv")
    Timestamp = pd.to_datetime(data["timestamp_(min)"])
    data["Timestamp"] = Timestamp
    data = data.set_index("Timestamp")
    labels = pd.read_csv("Data/input/PSM/test_label.csv")
    labels = labels.iloc[:,1].values
    data = data.astype(float)
    
    #%%
    
    feature = data.iloc[:,:25]
    scaler = StandardScaler()
    

    norm_feature = scaler.fit_transform(feature)

    n_sensor = norm_feature.shape[1]

    norm_feature = pd.DataFrame(norm_feature, columns= data.columns[1:], index = Timestamp)
    norm_feature = norm_feature.dropna(axis=1)
    train_df = norm_feature.iloc[:int(train_split*len(data))]
    train_label = labels[:int(train_split*len(data))]

    val_df = norm_feature.iloc[int(0.6*len(data)):int(0.8*len(data))]
    val_label = labels[int(0.6*len(data)):int(0.8*len(data))]
    

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

def loader_PSM_OCC(root, batch_size, window_size, stride_size,train_split,label=False):
    
    data = pd.read_csv("Data/input/PSM/train.csv")
    Timestamp = pd.to_datetime(data["timestamp_(min)"])
    data["Timestamp"] = Timestamp
    data = data.set_index("Timestamp")
    
    labels = [0] * len(data)
    data = data.astype(float)
    
    #%%
    print(data.shape)
    feature = data.iloc[:,:25]
    print(feature.shape)


    scaler = StandardScaler()
    norm_feature = scaler.fit_transform(feature)

    n_sensor = norm_feature.shape[1]


    norm_feature = pd.DataFrame(norm_feature, columns= data.columns[1:], index = Timestamp)

    norm_feature = norm_feature.dropna(axis=0)

 
    train_df = norm_feature.iloc[:]
    train_label = labels[:]
    print('trainset size',train_df.shape, 'anomaly ration', sum(train_label)/len(train_label))

    val_df = norm_feature.iloc[int(train_split*len(data)):]
    val_label = labels[int(train_split*len(data)):]
    
    
    
    
    data = pd.read_csv("Data/input/PSM/test.csv")
    Timestamp = pd.to_datetime(data["timestamp_(min)"])
    data["Timestamp"] = Timestamp
    data = data.set_index("Timestamp")
    labels = pd.read_csv("Data/input/PSM/test_label.csv")
    labels = labels.iloc[:,1].values
    data = data.astype(float)
    
    #%%
    
    feature = data.iloc[:,:25]
    
   
    scaler = StandardScaler()
    norm_feature = scaler.fit_transform(feature)

    n_sensor = norm_feature.shape[1]
 
    norm_feature = pd.DataFrame(norm_feature, columns= data.columns[1:], index = Timestamp)
    norm_feature = norm_feature.dropna(axis=1)
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


class SWat_dataset(Dataset):
    def __init__(self, df, label, window_size=60, stride_size=10) -> None:
        super(SWat_dataset, self).__init__()
        self.df = df
        self.window_size = window_size
        self.stride_size = stride_size

        self.data, self.idx, self.label = self.preprocess(df,label)
        self.columns = np.append(df.columns, ["Label"])
        self.timeindex = df.index[self.idx]
        print('label', self.label.shape, sum(self.label)/len(self.label))
        print('idx',self.idx.shape)
        print('data',self.data.shape)
        # print(len(self.data), len(self.idx), len(self.label))
    def preprocess(self, df, label):

        start_idx = np.arange(0,len(df)-self.window_size,self.stride_size)
        end_idx = np.arange(self.window_size, len(df), self.stride_size)
        
    
        label = [0 if sum(label[index:index+self.window_size]) == 0 else 1 for index in start_idx]
        return df.values, start_idx, np.array(label)

    def __len__(self):

        length = len(self.idx)

        return length

    def __getitem__(self, index):
        #  N X K X L X D 
        """
        """
        # print(self.window_size)
        start = self.idx[index]
        end = start + self.window_size
        data = self.data[start:end].reshape([self.window_size,-1, 1])
        # print('shape',data.shape)
        return torch.FloatTensor(data).transpose(0,1), self.label[index], index
