#%%
from math import gamma
import os
import argparse
from statistics import mode
import torch
from models.DeepSAD import DeepSVDD,DeepSAD
from models.DROCC import DROCCTrainer, LSTM_FC #DROCC
from models.GAN import R_Net, D_Net, CNNAE, train_model, R_Loss, D_Loss, test_single_epoch
import numpy as np
from utils import ROC

# from data import fetch_dataloaders


parser = argparse.ArgumentParser()
# files
parser.add_argument('--data_dir', type=str, 
                    default='Data/input/SWaT_Dataset_Attack_v0.csv', help='Location of datasets.')
parser.add_argument('--output_dir', type=str, 
                    default='./checkpoint/model')
parser.add_argument('--name',default='GANF_Water')
# restore
parser.add_argument('--graph', type=str, default='None')
parser.add_argument('--model', type=str, choices = ['DeepSVDD', 'DeepSAD', 'DROCC', 'EncDecAD', 'ALOCC'] ,default='None')
parser.add_argument('--seed', type=int, default=18, help='Random seed to use.')
parser.add_argument('--load', type=str, default="")

# made parameters
parser.add_argument('--n_blocks', type=int, default=1, help='Number of blocks to stack in a model (MADE in MAF; Coupling+BN in RealNVP).')
parser.add_argument('--n_components', type=int, default=1, help='Number of Gaussian clusters for mixture of gaussians models.')
parser.add_argument('--hidden_size', type=int, default=32, help='Hidden layer size for MADE (and each MADE block in an MAF).')
parser.add_argument('--n_hidden', type=int, default=1, help='Number of hidden layers in each MADE.')
parser.add_argument('--batch_norm', type=bool, default=False)
# training params
parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--n_epochs', type=int, default=3)
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate.')
parser.add_argument('--log_interval', type=int, default=1, help='How often to show loss statistics and save samples.')


args = parser.parse_known_args()[0]
args.cuda = torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")



import random
import numpy as np
import math
#%%



for seed in range(15,20):
 
    args.seed = seed
    print(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        
    from Dataset import load_smd_smap_msl, loader_SWat, loader_WADI, loader_PSM, loader_WADI_OCC

    if args.name == 'SWaT':
        train_loader, val_loader, test_loader, n_sensor = loader_SWat(args.data_dir, \
                                                                        args.batch_size, args.window_size, args.stride_size, args.train_split)

    elif args.name == 'Wadi':
        train_loader, val_loader, test_loader, n_sensor = loader_WADI(args.data_dir, \
                                                                    args.batch_size, args.window_size, args.stride_size, args.train_split)

    elif args.name == 'SMAP' or args.name == 'MSL' or args.name.startswith('machine'):
        train_loader, val_loader, test_loader, n_sensor = load_smd_smap_msl(args.name, \
                                                                    args.batch_size, args.window_size, args.stride_size, args.train_split)

    elif args.name == 'PSM':
        train_loader, val_loader, test_loader, n_sensor = loader_PSM(args.name, \
                                                                    args.batch_size, args.window_size, args.stride_size, args.train_split)
    print("Loading dataset")
    print(args.name)
    

    if args.model == 'DeepSVDD':
        model = DeepSVDD(n_sensor, size, device)

        if args.load:
            model.ae_net.encoder.load_state_dict(torch.load(args.load)['model'])
            c = torch.load(args.load)['c']
        model.train(train_loader, test_loader, args, device)
        gt, pre = model.test(test_loader, c,1, device)
        ROC(args, gt, pre)
    
    elif args.model == 'DeepSAD':
        model = DeepSAD(n_sensor, size, device)

        if args.load:
            model.ae_net.encoder.load_state_dict(torch.load(args.load)['model'])
            c = torch.load(args.load)['c']
        model.train(train_loader, test_loader, args, device)
        gt, pre = model.test(test_loader, c,1, device)
        ROC(args, gt, pre)
        
    elif args.model == 'DROCC':
        net = LSTM_FC(input_dim=size).to(device)
        optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20], gamma=0.1)
        radius = math.sqrt(size)/2
        gamma = 2
        lam = 0.0001


        if args.load:
            net.load_state_dict(torch.load(args.load)['model'])
        model = DROCCTrainer(net, optimizer, lam, radius, gamma, device)
        model.train(args, seed, train_loader=train_loader, test_loader=test_loader,lr_scheduler=scheduler, total_epochs=40, save_path='./othermodel', name='DROCC' )
        gt, pre = model.test(test_loader)
        ROC(args, gt, pre)
    elif args.model == 'ALOCC':
        model1 = R_Net(in_channels=size, n_channels=n_channels)
        model2 = D_Net(in_resolution=60, in_channels=size, n_channels=n_channels)
        model1 = model1.to(device)
        model2 = model2.to(device)

        if args.load:
            model1.load_state_dict(torch.load(args.load)['r_net'])
            model2.load_state_dict(torch.load(args.load)['d_net'])
        train_model(args, model1, model2, train_loader= train_loader, test_loader=test_loader)
        gt, pre = test_single_epoch(model1, model2, R_Loss, D_Loss, test_loader, device)
        ROC(args, gt, pre)
    
