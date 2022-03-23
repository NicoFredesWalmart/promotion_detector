import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from scipy.special import logit, expit
## modes to train c
noTrain = 0
train = 1


class PromoDetecter(torch.nn.Module):

    def __init__(self, num_stds, threshold, percentile, device, lm=1.0, train_hyp=train, train_net=noTrain, max_days=30):
        super(PromoDetecter, self).__init__()

        #model parameters
        
        #self.win_size = win_size

        self.device = device
        
        #Value greater than 0
        self.num_stds = num_stds
        
        #Value between 0 and 1
        self.threshold = threshold

        #Value between 0 and 1
        self.percentile = percentile
        
        #training parameters
        self.lm = lm

        self.train_hyp = train_hyp

        self.train_net = train_net

        self.max_days = max_days

        #activation function
        self.sig = nn.Sigmoid()

        self.R = nn.ReLU()

        #exponential train parametera

        if self.train_hyp == train:
            self.num_stds = np.log(self.num_stds / self.lm)
            self.num_stds = nn.Parameter(torch.Tensor([self.num_stds, ]))

            self.threshold = logit(self.threshold / self.lm)
            self.threshold = nn.Parameter(torch.Tensor([self.threshold, ]))

            self.percentile = logit(self.percentile / self.lm)
            self.percentile = nn.Parameter(torch.Tensor([self.percentile, ]))

            self.lm = torch.as_tensor(self.lm)

        elif self.train_net == train:
            self.linear1 = nn.Linear(self.max_days, self.max_days)
            self.linear2 = nn.Linear(2, 1)

    def forward(self, x, x0, l, l0, pad, pad0):

        #x va a ser de dimensiones batch size (barcodes) X dias (precios)
        if self.train_hyp == train:
            num_stds = torch.exp(self.num_stds * self.lm)
            threshold = self.sig(self.threshold * self.lm)
            percentile = self.sig(self.percentile * self.lm)

        else:
            num_stds = self.num_stds
            threshold = self.threshold
            percentile = self.percentile

        y = self.predict(x, x0, l, pad, pad0, num_stds, threshold, percentile)

        if self.train_net == noTrain:
            return y
        else:
            x0 = pad0*(x0 - self.get_mean(x0, l0))
            max_elements, max_idxs = torch.max(torch.abs(x0), dim=1)
            x0 = x0/(max_elements.reshape(x0.shape[0], 1) + 1e-12)
            x0 = self.linear1(x0)
            x0 = F.dropout(self.sig(x0), 0.1)
            y = self.sig(y)
            x0 = x0.reshape(x0.shape[0], self.max_days, 1)
            y = y.reshape(x0.shape[0], self.max_days, 1)         
            y = torch.cat((x0, y), 2).reshape(x0.shape[0], self.max_days, 2)
            y = self.linear2(y)
            y = y.reshape(x0.shape[0], self.max_days)
            return y
            

    def predict(self, x, x0, l, pad, pad0, num_stds, threshold, percentile):
        '''
        Performs the prediction by aggregating the predictions given by
        each of the heuristics. The aggregation method is a simple majo
        rity vote.
        '''

        h1 = self.is_promo_max(x, x0, l, pad, pad0, threshold)
        h2 = self.is_promo_order(x, x0, l, pad, pad0, percentile)
        h3 = self.is_promo_mean(x, x0, l, pad, pad0, num_stds)

        #flag = (h1 + h2 + h3) > 1.5
        flag = pad0*(h1 + h2 + h3 - 1.5)
        return flag

    def is_promo_max(self, x, x0, l, pad, pad0, threshold):
        '''
        Heuristic 1: predicts promo if observation is far from the max

        Given the reference price distribution and the threshold parameter,
        if the price we are observing is below the maximum of the reference
        prices times the threshold, we flag the current observed price as
        promotional.
        '''
        max_elements, max_idxs = torch.max(x, dim=1)
        #flag = (x/max_elements.reshape(x.shape[0], 1)) < threshold
        flag =  pad0 * self.sig(threshold - (x0/max_elements.reshape(x.shape[0], 1)))
        return flag

    def is_promo_order(self, x, x0, l, pad, pad0, percentile):
        '''
        Heuristic 2: predicts promo if observation is low in the ordered
        list of prices.

        Given the reference price distribution and the percentile parameter
        if the price we are observing is below the percentile then it is
        flagged as promotional.
        '''
        srtd, idx = torch.sort(x, 1)

        uno = torch.ones(x.shape[0], 1).to(self.device).type(torch.int64)

        n = (l * percentile).type(torch.float64)
        #n = ((n >= 0) * ((n < 1) + (n >= 1) * n)).type(torch.float64)
        n = self.R(self.R(n - uno.type(torch.float64)) + uno.type(torch.float64))

        alpha = n % 1

        cut_off = x.shape[1] - l + n.type(torch.int64) - uno

        indices = cut_off.reshape(x.shape[0]).type(torch.int64)
        I = torch.eye(x.shape[0], device=self.device)
        #revisar
        value0 = torch.sum(torch.index_select(srtd, 1, indices)*I, 1).reshape(x.shape[0], 1)
        indices2 = indices + uno.reshape(x.shape[0])
        max = srtd.shape[1] - 1
        #indices2 = ((indices2 >= 0) * ((indices2 < max) * indices2 + (indices2 >= max) * max)).type(torch.int64)
        indices2 = (self.R(indices2) - self.R(indices2 - max)).type(torch.int64)
        value1 = torch.sum(torch.index_select(srtd, 1,  indices2) * I, 1).reshape(x.shape[0], 1)

        corte = (uno - alpha)*value0 + alpha*value1
        flag = pad0*self.sig(corte - x0)
        return flag

    def is_promo_mean(self, x, x0, l, pad, pad0, num_stds):
        '''
        Heuristic 3: predicts promo if observation is far from the mean

        Given the reference price distribution and the num_stds parameter,
        if the price we are observing is below the mean of the distribution
        minus the standard deviation times num_stds, then we flag the price
        as promotional.
        '''
        mean = self.get_mean(x, l)
        std = self.get_std(x, l, pad)
        #flag = x < (mean - num_stds*std)
        flag = pad0*self.sig(mean - num_stds*std - x0)
        return flag

    def get_mean(self, x, l):
        suma = torch.sum(x, 1).reshape(x.shape[0], 1)
        mean = (suma/l).reshape(x.shape[0], 1)
        return mean

    def get_std(self, x, l, pad):

        mean = self.get_mean(x, l)
        std = torch.sqrt(self.get_mean(pad*torch.abs(x - mean)**2, l))
        return std

    def get_real_values(self):

        if self.train_hyp == noTrain:
            return self.num_stds, self.threshold, self.percentile
        elif self.train_hyp == train:
            num_stds = np.exp(self.num_stds.item() * self.lm.item())
            #usar numpy sigmoide ?
            threshold = self.sig(self.threshold * self.lm).item()
            percentile = self.sig(self.percentile * self.lm).item()
            return num_stds, threshold, percentile

