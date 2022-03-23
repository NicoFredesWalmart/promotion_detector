import json
import os
import os.path as op
import pickle
import collections
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
    

class Dataset(Dataset):
    
    def __init__(self, path_dataset, win_size):
        super().__init__()
        #self.dataset = load_data(fname, config)
        self.path_dataset = path_dataset
        self.dataset = os.listdir(path_dataset)
        self.win_size = win_size

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):

        if torch.is_tensor(index):
            index = index.tolist()
        
        sample = torch.load(op.join(self.path_dataset, self.dataset[index]))
        sample_0 = torch.load(op.join(self.path_dataset, self.dataset[index]))
        barcode = self.dataset[index]
        try:
            n_prices = len(sample.squeeze().tolist())
        except:
            n_prices = 1
        sample = sample.reshape(n_prices)
        #sample_0 = sample
        pad0 = torch.ones(n_prices)

        if n_prices > self.win_size:
            sample = sample[:self.win_size]
        
        return sample, sample_0, barcode, pad0, n_prices


def generate_batch(batch):
    samples = []
    samples_0 = []
    lens = []
    l0 = []
    pad0 = []
    barcodes = [] 
    #escoger los elementos a usar en la red, se puede agregar el input 2d
    for price_promo in batch:
        samples.append(price_promo[0])
        samples_0.append(price_promo[1])
        l0.append(price_promo[4])
        pad0.append(price_promo[3])
        try:
            lens.append(len(price_promo[0].squeeze().tolist()))
        except:
            lens.append(1)
        barcodes.append(price_promo[2])
    #se hace padding hasta el largo de la prote mas larga del batch
    x = pad_sequence(samples, padding_value=0.0)
    x0 = pad_sequence(samples_0, padding_value=0.0)

    #calcular padding mask haciendo padding con otra cosa
    pad_ones = pad_sequence(samples, padding_value=1.0)
    #esto debe ser de dimensiones batch_size x prox_max_len
    pad = (x != pad_ones)
    
    l = torch.tensor(lens).reshape(len(lens), 1)
    l0 = torch.tensor(l0).reshape(len(l0), 1)
    
    pad0 = pad_sequence(pad0, padding_value=0.0)
    #x, pad, 
    return x.T, x0.T, l, l0, pad.T, pad0.T, barcodes

