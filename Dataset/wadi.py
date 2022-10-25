import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd
import numpy as np
from datetime import datetime

def loader_WADI_OCC(root, batch_size, window_size, stride_size, train_split,label=False):

    data = pd.read_csv("Data/input/WADI_14days.csv",sep=",")#, nrows=1000)
    labels=[]
    Timestamp = pd.to_datetime(data['Date'] + ' ' + data['Time'])
    data=data.drop(data.columns[[0,1,2,50,51,86,87]],axis=1) # Drop the empty and date/time columns
    labels = [0]*len(data)

    data = data.astype(float)
    n_sensor = len(data.columns)

    print('sensor',n_sensor)
    #%%
    feature = data

    min_scaler = StandardScaler()
    feature = min_scaler.fit_transform(feature)
    

    norm_feature = pd.DataFrame(feature, index = Timestamp, columns=data.columns)

    norm_feature = norm_feature.dropna(axis=0)


    train_df = norm_feature.iloc[:]
    train_label = labels[:]
    print('trainset size',train_df.shape, 'anomaly ration', sum(train_label)/len(train_label))
  
   
    val_df = norm_feature.iloc[int(train_split*len(data)):]
    val_label = labels[int(train_split*len(data)):]
    data = pd.read_csv("Data/input/WADI_attackdata.csv",sep=",")#, nrows=1000)
    labels=[]

    # attack.reset_index()
    for index, row in data.iterrows():
        date_temp=row['Date']
        date_mask="%m/%d/%Y"
        date_obj=datetime.strptime(date_temp, date_mask)
        time_temp=row['Time']
        time_mask="%I:%M:%S.%f %p"
        time_obj=datetime.strptime(time_temp,time_mask)

        if date_obj==datetime.strptime('10/9/2017', '%m/%d/%Y'):
            if time_obj>=datetime.strptime('7:25:00.000 PM', '%I:%M:%S.%f %p') and time_obj<=datetime.strptime('7:50:16.000 PM', '%I:%M:%S.%f %p'):
                labels.append('Attack')
                continue

        if date_obj==datetime.strptime('10/10/2017', '%m/%d/%Y'):
            if time_obj>=datetime.strptime('10:24:10.000 AM', '%I:%M:%S.%f %p') and time_obj<=datetime.strptime('10:34:00.000 AM', '%I:%M:%S.%f %p'):
                labels.append('Attack')
                continue
            elif time_obj>=datetime.strptime('10:55:00.000 AM', '%I:%M:%S.%f %p') and time_obj<=datetime.strptime('11:24:00.000 AM', '%I:%M:%S.%f %p'):
                labels.append('Attack')
                continue
            elif time_obj>=datetime.strptime('11:30:40.000 AM', '%I:%M:%S.%f %p') and time_obj<=datetime.strptime('11:44:50.000 AM', '%I:%M:%S.%f %p'):
                labels.append('Attack')
                continue
            elif time_obj>=datetime.strptime('1:39:30.000 PM', '%I:%M:%S.%f %p') and time_obj<=datetime.strptime('1:50:40.000 PM', '%I:%M:%S.%f %p'):
                labels.append('Attack')
                continue
            elif time_obj>=datetime.strptime('2:48:17.000 PM', '%I:%M:%S.%f %p') and time_obj<=datetime.strptime('2:59:55.000 PM', '%I:%M:%S.%f %p'):
                labels.append('Attack')
                continue
            elif time_obj>=datetime.strptime('5:40:00.000 PM', '%I:%M:%S.%f %p') and time_obj<=datetime.strptime('5:49:40.000 PM', '%I:%M:%S.%f %p'):
                labels.append('Attack')
                continue
            elif time_obj>=datetime.strptime('10:55:00.000 AM', '%I:%M:%S.%f %p') and time_obj<=datetime.strptime('10:56:27.000 AM', '%I:%M:%S.%f %p'):
                labels.append('Attack')
                continue
        
        if date_obj==datetime.strptime('10/11/2017', '%m/%d/%Y'):
            if time_obj>=datetime.strptime('11:17:54.000 AM', '%I:%M:%S.%f %p') and time_obj<=datetime.strptime('11:31:20.000 AM', '%I:%M:%S.%f %p'):
                labels.append('Attack')
                continue
            elif time_obj>=datetime.strptime('11:36:31.000 AM', '%I:%M:%S.%f %p') and time_obj<=datetime.strptime('11:47:00.000 AM', '%I:%M:%S.%f %p'):
                labels.append('Attack')
                continue
            elif time_obj>=datetime.strptime('11:59:00.000 AM', '%I:%M:%S.%f %p') and time_obj<=datetime.strptime('12:05:00.000 PM', '%I:%M:%S.%f %p'):
                labels.append('Attack')
                continue
            elif time_obj>=datetime.strptime('12:07:30.000 PM', '%I:%M:%S.%f %p') and time_obj<=datetime.strptime('12:10:52.000 PM', '%I:%M:%S.%f %p'):
                labels.append('Attack')
                continue
            elif time_obj>=datetime.strptime('12:16:00.000 PM', '%I:%M:%S.%f %p') and time_obj<=datetime.strptime('12:25:36.000 PM', '%I:%M:%S.%f %p'):
                labels.append('Attack')
                continue
            elif time_obj>=datetime.strptime('3:26:30.000 PM', '%I:%M:%S.%f %p') and time_obj<=datetime.strptime('3:37:00.000 PM', '%I:%M:%S.%f %p'):
                labels.append('Attack')
                continue

        labels.append('Normal')

    Timestamp = pd.to_datetime(data['Date'] + ' ' + data['Time'])
    data=data.drop(data.columns[[0,1,2,50,51,86,87]],axis=1) # Drop the empty and date/time columns
    labels = [ int(l!= 'Normal' ) for l in labels]
    data = data.astype(float)
    n_sensor = len(data.columns)

    #%%
    feature = data
    min_scaler = StandardScaler()
    feature = min_scaler.fit_transform(feature)
    

    norm_feature = pd.DataFrame(feature, index = Timestamp, columns=data.columns)

    norm_feature = norm_feature.dropna(axis=1)

    test_df = norm_feature.iloc[int(train_split*len(data)):]
    test_label = labels[int(train_split*len(data)):]
    print('testset size',test_df.shape, 'anomaly ration', sum(test_label)/len(test_label))

    if label:
        train_loader = DataLoader(WADI_dataset(train_df,train_label, window_size, stride_size), batch_size=batch_size, shuffle=False)
    else:
        train_loader = DataLoader(WADI_dataset(train_df,train_label, window_size, stride_size), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(WADI_dataset(val_df,val_label, window_size, stride_size), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(WADI_dataset(test_df,test_label, window_size, stride_size), batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, n_sensor



def loader_WADI(root, batch_size, window_size, stride_size,train_split,label=False):
    
    
    
    
    data = pd.read_csv("Data/input/WADI_attackdata.csv",sep=",")#, nrows=1000)
    labels=[]


    for index, row in data.iterrows():
        date_temp=row['Date']
        date_mask="%m/%d/%Y"
        date_obj=datetime.strptime(date_temp, date_mask)
        time_temp=row['Time']
        time_mask="%I:%M:%S.%f %p"
        time_obj=datetime.strptime(time_temp,time_mask)

        if date_obj==datetime.strptime('10/9/2017', '%m/%d/%Y'):
            if time_obj>=datetime.strptime('7:25:00.000 PM', '%I:%M:%S.%f %p') and time_obj<=datetime.strptime('7:50:16.000 PM', '%I:%M:%S.%f %p'):
                labels.append('Attack')
                continue

        if date_obj==datetime.strptime('10/10/2017', '%m/%d/%Y'):
            if time_obj>=datetime.strptime('10:24:10.000 AM', '%I:%M:%S.%f %p') and time_obj<=datetime.strptime('10:34:00.000 AM', '%I:%M:%S.%f %p'):
                labels.append('Attack')
                continue
            elif time_obj>=datetime.strptime('10:55:00.000 AM', '%I:%M:%S.%f %p') and time_obj<=datetime.strptime('11:24:00.000 AM', '%I:%M:%S.%f %p'):
                labels.append('Attack')
                continue
            elif time_obj>=datetime.strptime('11:30:40.000 AM', '%I:%M:%S.%f %p') and time_obj<=datetime.strptime('11:44:50.000 AM', '%I:%M:%S.%f %p'):
                labels.append('Attack')
                continue
            elif time_obj>=datetime.strptime('1:39:30.000 PM', '%I:%M:%S.%f %p') and time_obj<=datetime.strptime('1:50:40.000 PM', '%I:%M:%S.%f %p'):
                labels.append('Attack')
                continue
            elif time_obj>=datetime.strptime('2:48:17.000 PM', '%I:%M:%S.%f %p') and time_obj<=datetime.strptime('2:59:55.000 PM', '%I:%M:%S.%f %p'):
                labels.append('Attack')
                continue
            elif time_obj>=datetime.strptime('5:40:00.000 PM', '%I:%M:%S.%f %p') and time_obj<=datetime.strptime('5:49:40.000 PM', '%I:%M:%S.%f %p'):
                labels.append('Attack')
                continue
            elif time_obj>=datetime.strptime('10:55:00.000 AM', '%I:%M:%S.%f %p') and time_obj<=datetime.strptime('10:56:27.000 AM', '%I:%M:%S.%f %p'):
                labels.append('Attack')
                continue
        
        if date_obj==datetime.strptime('10/11/2017', '%m/%d/%Y'):
            if time_obj>=datetime.strptime('11:17:54.000 AM', '%I:%M:%S.%f %p') and time_obj<=datetime.strptime('11:31:20.000 AM', '%I:%M:%S.%f %p'):
                labels.append('Attack')
                continue
            elif time_obj>=datetime.strptime('11:36:31.000 AM', '%I:%M:%S.%f %p') and time_obj<=datetime.strptime('11:47:00.000 AM', '%I:%M:%S.%f %p'):
                labels.append('Attack')
                continue
            elif time_obj>=datetime.strptime('11:59:00.000 AM', '%I:%M:%S.%f %p') and time_obj<=datetime.strptime('12:05:00.000 PM', '%I:%M:%S.%f %p'):
                labels.append('Attack')
                continue
            elif time_obj>=datetime.strptime('12:07:30.000 PM', '%I:%M:%S.%f %p') and time_obj<=datetime.strptime('12:10:52.000 PM', '%I:%M:%S.%f %p'):
                labels.append('Attack')
                continue
            elif time_obj>=datetime.strptime('12:16:00.000 PM', '%I:%M:%S.%f %p') and time_obj<=datetime.strptime('12:25:36.000 PM', '%I:%M:%S.%f %p'):
                labels.append('Attack')
                continue
            elif time_obj>=datetime.strptime('3:26:30.000 PM', '%I:%M:%S.%f %p') and time_obj<=datetime.strptime('3:37:00.000 PM', '%I:%M:%S.%f %p'):
                labels.append('Attack')
                continue

        labels.append('Normal')
 
    Timestamp = pd.to_datetime(data['Date'] + ' ' + data['Time'])
    data=data.drop(data.columns[[0,1,2,50,51,86,87]],axis=1) # Drop the empty and date/time columns
    labels = [ int(l!= 'Normal' ) for l in labels]

    data = data.astype(float)

    n_sensor = len(data.columns)

    print('sensor',n_sensor)

    feature = data
    scaler = StandardScaler()
    norm_feature = scaler.fit_transform(feature)
    norm_feature = pd.DataFrame(norm_feature, index = Timestamp, columns=data.columns)
    norm_feature = norm_feature.dropna(axis=0)


    train_df = norm_feature.iloc[:int(train_split*len(data))]
    train_label = labels[:int(train_split*len(data))]
    print('trainset size',train_df.shape, 'anomaly ration', sum(train_label)/len(train_label))

    val_df = norm_feature.iloc[int(0.6*len(data)):int(0.8*len(data))]
    val_label = labels[int(0.6*len(data)):int(0.8*len(data))]

    test_df = norm_feature.iloc[int(train_split*len(data)):]
    test_label = labels[int(train_split*len(data)):]
    print('testset size',test_df.shape, 'anomaly ration', sum(test_label)/len(test_label))

    if label:
        train_loader = DataLoader(WADI_dataset(train_df,train_label, window_size, stride_size), batch_size=batch_size, shuffle=False)
    else:
        train_loader = DataLoader(WADI_dataset(train_df,train_label, window_size, stride_size), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(WADI_dataset(val_df,val_label, window_size, stride_size), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(WADI_dataset(test_df,test_label, window_size, stride_size), batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, n_sensor

class WADI_dataset(Dataset):
    def __init__(self, df, label, window_size=60, stride_size=10) -> None:
        super(WADI_dataset, self).__init__()
        self.df = df
        self.window_size = window_size
        self.stride_size = stride_size

        self.data, self.idx, self.label = self.preprocess(df,label)
        self.columns = np.append(df.columns, ["Label"])
        self.timeindex = df.index[self.idx]
        print('label', self.label.shape, sum(self.label)/len(self.label))
        print('idx',self.idx.shape)
        print('data',self.data.shape)

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