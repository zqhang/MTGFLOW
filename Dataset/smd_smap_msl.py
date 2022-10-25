# -*- coding: utf-8 -*-
import os
import pickle
import torch
import numpy as np

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils.data import Dataset, DataLoader
import pandas as pd

prefix = "Data/input/processed"


def save_z(z, filename='z'):
    """
    save the sampled z in a txt file
    """
    for i in range(0, z.shape[1], 20):
        with open(filename + '_' + str(i) + '.txt', 'w') as file:
            for j in range(0, z.shape[0]):
                for k in range(0, z.shape[2]):
                    file.write('%f ' % (z[j][i][k]))
                file.write('\n')
    i = z.shape[1] - 1
    with open(filename + '_' + str(i) + '.txt', 'w') as file:
        for j in range(0, z.shape[0]):
            for k in range(0, z.shape[2]):
                file.write('%f ' % (z[j][i][k]))
            file.write('\n')


def get_data_dim(dataset):
    if dataset == 'SMAP':
        return 25
    elif dataset == 'MSL':
        return 55
    elif str(dataset).startswith('machine'):
        return 38
    else:
        raise ValueError('unknown dataset '+str(dataset))


def load_smd_smap_msl(dataset, batch_size = 512, window_size = 60, stride_size = 10, train_split = 0.6, label=False, do_preprocess=True, train_start=0,
             test_start=0):
    """
    get data from pkl files

    return shape: (([train_size, x_dim], [train_size] or None), ([test_size, x_dim], [test_size]))
    """
   
    x_dim = get_data_dim(dataset)
 
    try:
        f = open(os.path.join(prefix, dataset + '_test.pkl'), "rb")
        test_data = pickle.load(f).reshape((-1, x_dim))[test_start:, :]
        f.close()
    except (KeyError, FileNotFoundError):
        test_data = None
    try:
        f = open(os.path.join(prefix, dataset + "_test_label.pkl"), "rb")
        test_label = pickle.load(f).reshape((-1))[test_start:]
        f.close()
    except (KeyError, FileNotFoundError):
        test_label = None
    print('testset size',test_label.shape, 'anomaly ration', sum(test_label)/len(test_label))

    whole_data = test_data
    whole_label = test_label
    print('testset size',whole_label.shape, 'anomaly ration', sum(whole_label)/len(whole_label))
    if do_preprocess:
        whole_data = preprocess(whole_data)
   
    n_sensor = whole_data.shape[1]
    print('n_sensor', n_sensor)

    train_df = whole_data[:int(train_split*len(whole_data))]
    train_label = whole_label[:int(train_split*len(whole_data))]

    val_df = whole_data[int(0.6*len(whole_data)):int(0.8*len(whole_data))]
    val_label = whole_label[int(0.6*len(whole_data)):int(0.8*len(whole_data))]

    test_df = whole_data[int(train_split*len(whole_data)):]
    test_label = whole_label[int(train_split*len(whole_data)):]

    print('train size',train_label.shape, 'anomaly ration', sum(train_label)/len(train_label))
    print('test size',test_label.shape, 'anomaly ration', sum(test_label)/len(test_label))


    if label:
        train_loader = DataLoader(Smd_smap_msl_dataset(train_df,train_label, window_size, stride_size), batch_size=batch_size, shuffle=False)
    else:
        train_loader = DataLoader(Smd_smap_msl_dataset(train_df,train_label, window_size, stride_size), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(Smd_smap_msl_dataset(val_df,val_label, window_size, stride_size), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(Smd_smap_msl_dataset(test_df,test_label, window_size, stride_size), batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader, n_sensor




def load_smd_smap_msl_occ(dataset, batch_size = 512, window_size = 60, stride_size = 10, train_split = 0.6, label=False, do_preprocess=True, train_start=0,
             test_start=0):
    """
    get data from pkl files

    return shape: (([train_size, x_dim], [train_size] or None), ([test_size, x_dim], [test_size]))
    """
 
    x_dim = get_data_dim(dataset)
    f = open(os.path.join(prefix, dataset + '_train.pkl'), "rb")
    train_data = pickle.load(f).reshape((-1, x_dim))[train_start:, :]

    f.close()
    try:
        f = open(os.path.join(prefix, dataset + '_test.pkl'), "rb")
        test_data = pickle.load(f).reshape((-1, x_dim))[test_start:, :]
        f.close()
    except (KeyError, FileNotFoundError):
        test_data = None
    try:
        f = open(os.path.join(prefix, dataset + "_test_label.pkl"), "rb")
        test_label = pickle.load(f).reshape((-1))[test_start:]
        f.close()
    except (KeyError, FileNotFoundError):
        test_label = None

  
    if do_preprocess:
        train_data = preprocess(train_data)
        test_data = preprocess(test_data)

    print("train set shape: ", train_data.shape)
    print("test set shape: ", test_data.shape)
    print("test set label shape: ", test_label.shape)
    n_sensor = train_data.shape[1]
    print('n_sensor', n_sensor)

    train_df = train_data[:]
    train_label = [0]*len(train_df)

    val_df = train_data[int(train_split*len(train_data)):]
    val_label = [0]*len(val_df)

  
    test_df = test_data[int(train_split*len(test_data)):]
    test_label = test_label[int(train_split*len(test_data)):]
    print('testset size',test_label.shape, 'anomaly ration', sum(test_label)/len(test_label))

    if label:
        train_loader = DataLoader(Smd_smap_msl_dataset(train_df,train_label, window_size, stride_size), batch_size=batch_size, shuffle=False)
    else:
        train_loader = DataLoader(Smd_smap_msl_dataset(train_df,train_label, window_size, stride_size), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(Smd_smap_msl_dataset(val_df,val_label, window_size, stride_size), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(Smd_smap_msl_dataset(test_df,test_label, window_size, stride_size), batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader, n_sensor



def preprocess(df, mode = 'Normal'):
    """returns normalized and standardized data.
    """

    df = np.asarray(df, dtype=np.float32)

    if len(df.shape) == 1:
        raise ValueError('Data must be a 2-D array')

    if np.any(sum(np.isnan(df)) != 0):
        print('Data contains null values. Will be replaced with 0')
        df = np.nan_to_num()

    # normalize data
    if mode == 'Normal':
        df = StandardScaler().fit_transform(df)
    else:
        df = MinMaxScaler().fit_transform(df)
    print('Data normalized')

    return df


class Smd_smap_msl_dataset(Dataset):
    def __init__(self, df, label, window_size=60, stride_size=10) -> None:
        super(Smd_smap_msl_dataset, self).__init__()
        self.df = df
        self.window_size = window_size
        self.stride_size = stride_size

        self.data, self.idx, self.label = self.preprocess(df,label)
        # self.columns = np.append(df.columns, ["Label"])
        # self.timeindex = df.index[self.idx]
        print('label', self.label.shape, sum(self.label)/len(self.label))
        print('idx',self.idx.shape)
        print('data',self.data.shape)
        # print(len(self.data), len(self.idx), len(self.label))
    def preprocess(self, df, label):

        start_idx = np.arange(0,len(df)-self.window_size,self.stride_size)
        end_idx = np.arange(self.window_size, len(df), self.stride_size)
        
      
        label = [0 if sum(label[index:index+self.window_size]) == 0 else 1 for index in start_idx]
        return df, start_idx, np.array(label)

    def __len__(self):

        length = len(self.idx)

        return length   

    def __getitem__(self, index):
        #  N X K X L X D 

        start = self.idx[index]
        end = start + self.window_size
        data = self.data[start:end].reshape([self.window_size,-1, 1])
        return torch.FloatTensor(data).transpose(0,1), self.label[index], index
