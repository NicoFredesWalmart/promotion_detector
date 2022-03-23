import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
import argparse
import os
import logging
from torch.utils.data import DataLoader
from loaddata import Dataset, generate_batch
import torch.backends.cudnn as cudnn
from pathlib import Path
import PromoDetecter
import numpy as np
import sys
import json
sys.path.append('../../')
import pickle
import pandas as pd

def predict(x, x0, l, l0, pad, pad0, labels, predictor_net, device, train=True):
    x = x.double().to(device)
    x0 = x0.double().to(device)
    l = l.double().to(device)
    l0 = l0.double().to(device)
    pad = pad.double().to(device)
    pad0 = pad0.double().to(device)
    labels = labels.double().to(device)

    # usar la red
    if train:
        prediction = predictor_net(x, x0, l, l0, pad, pad0)
    else:
        prediction = predictor_net(x, x0, l, l0, pad, pad0).detach()

    # arreglar las dimensiones cuando solo se carga una proteina
    if prediction.shape[0] == 1:
        labels = labels.unsqueeze(0)

    return prediction, labels



def val_epoch(dataloader, count_iter, predictor_net, writer, device, loss_fn):
    #valor para la funcion escalon cuando la entrada es 0
    losses = 0
    values = torch.tensor([1.0]).double().to(device)
    batches_number = len(dataloader)
    for i, (x, x0, l, l0, pad, pad0, labels) in enumerate(dataloader):

        prediction, labels = predict(x, x0, l, l0, pad, pad0, labels=labels,
                                     predictor_net=predictor_net, device=device,
                                     train=False)

        loss = loss_fn(prediction, labels)

        writer.add_scalar('Validation_Iteration_Loss', loss.item(), count_iter)
        losses += loss.item()

        count_iter += 1
        #sigmoide
        prediction = torch.sigmoid(prediction)
        #------calculos del data en general
        #round_prediction = torch.heaviside(prediction - 0.1, values)
        #save
        label_zeros = torch.zeros(labels.shape[0], 30)
        prediction_zeros = torch.zeros(prediction.shape[0], 30)

        label_zeros[:labels.shape[0], :labels.shape[1]] = labels
        prediction_zeros[:prediction.shape[0], :prediction.shape[1]] = prediction

        p0, p1 = prediction.shape[0], prediction.shape[1]
        la0, la1 = labels.shape[0], labels.shape[1]

        labels = label_zeros
        prediction = prediction_zeros

        label_pad = torch.zeros(la0, 30)
        prediction_pad = torch.zeros(p0, 30)

        label_pad[:la0, :la1] = torch.ones(la0, la1)
        prediction_pad[:p0, :p1] = torch.ones(p0, p1)
        

        if i == 0:
            labels_all = labels
            predictions_all = prediction
            label_zeros_all = label_pad
        else:
            labels_all = torch.cat((labels_all, labels), 0)
            predictions_all = torch.cat((predictions_all, prediction), 0)
            label_zeros_all = torch.cat((label_zeros_all, label_pad), 0)
    
    f_max = 0
    t_max = 0
    t_emin = 0
    error_min = 100
    error_fmax = 0

    predictions_all = predictions_all.to(device)
    labels_all = labels_all.to(device)
    label_zeros_all = label_zeros_all.to(device)

    for i in range(1,101):
        t = 0.01*i
        t = torch.tensor([t]).double().to(device)
        round_prediction = torch.heaviside((predictions_all - t).type(torch.float64), values)
        # false positive predicted per protein
        a = torch.sum(predictions_all)
        b = torch.sum(round_prediction)
        c = torch.sum(labels_all)
        condition = (labels_all - round_prediction == -1)

        d = torch.sum(condition)
        fp_protein = torch.sum(condition*label_zeros_all, 1)
        # false negative predicted per protein
        fn_protein = torch.sum((labels_all - round_prediction == 1)*label_zeros_all, 1)
        # real positive predicted per protein
        rp_protein = torch.sum((labels_all + round_prediction == 2)*label_zeros_all, 1)
        # total positive in label per protein
        p_protein = torch.sum((labels_all == 1)*label_zeros_all, 1)
        # precision final terms
        rp_p = 100 * (rp_protein + 1e-12) / (p_protein + 1e-12)
        # recall final terms
        rp_RpFp = 100 * (rp_protein + 1e-12) / (rp_protein + fp_protein + 1e-12)
        
        fake_positive = torch.sum(fp_protein).item()/fp_protein.shape[0]
        fake_negative = torch.sum(fn_protein).item()/fn_protein.shape[0]
        error = fake_positive + fake_negative

        #fake_positive = fake_positive.item() 
        #fake_negative = fake_negative.item()
        #error = error.item()
        t = t.item() 

        if error < error_min:
            error_min = error
            fake_negative_emin = fake_negative
            t_emin = t

        precession_t = torch.sum(rp_p).item()/rp_p.shape[0]
        recall_t = torch.sum(rp_RpFp).item()/rp_RpFp.shape[0]
        f = (2*precession_t*recall_t)/(precession_t + recall_t)
        
        #precession_t = precession_t.item()
        #recall_t = recall_t.item()
        #f = f.item()

        if f > f_max:
            fake_negative_fmax = fake_negative
            f_max = f
            t_max = t
            error_fmax = error

    epoch_loss = losses / batches_number
    return epoch_loss, count_iter, f_max, t_max, error_fmax, fake_negative_emin, t_emin, error_min, fake_negative_fmax


def train_epoch(dataloader, epoch, epochs_number, count_iter,
    predictor_net, optimizer, loss_fn, writer, device):
    losses = 0
    iter = 1
    batches_number = len(dataloader)
    for i, (x, x0, l, l0, pad, pad0, labels) in enumerate(dataloader):

        print('Epoch: ', epoch + 1, '/', epochs_number, ' Iteration: ', iter, '/',  batches_number)

        prediction, labels = predict(x, x0, l, l0, pad, pad0, labels=labels,
                                     predictor_net=predictor_net, device=device, train=True)

        optimizer.zero_grad()

        loss = loss_fn(prediction, labels)

        losses += loss.item()

        count_iter += 1
        iter += 1

        loss.backward()

        optimizer.step()

        num_stds, threshold, percentile = predictor_net.get_real_values()
        writer.add_scalar('Training_Iteration_Loss', loss.item(), count_iter)
        writer.add_scalar('hyperparameters/num_stds', num_stds, count_iter)
        writer.add_scalar('hyperparameters/threshold', threshold, count_iter)
        writer.add_scalar('hyperparameters/percentile', percentile, count_iter)
    epoch_loss = losses/batches_number
    return epoch_loss, count_iter

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Promotion Detector')

    parser.add_argument('-data_folder', '--data_folder', default="../data_train", type=str, help='folder with train and val data')
    parser.add_argument('-val_freq', '--validation_freq', default=1, type=int, help='number of train epoch for each validation epoch')
    parser.add_argument('-bs', '--batch_size', default=64, type=int, help='batch size')
    parser.add_argument('--root', type=str, default='../experiments/',
                        help='root of result file')
    parser.add_argument('--device', type=int, default=0,
                        help='index of devices where hgan is training')
    parser.add_argument('--epochs', type=int, default=4000, metavar='N',
                        help='number of epochs to train (default: 200)')
    parser.add_argument('--epochs_save', type=int, default=10,
                        help='number of epochs to saves models (default: 50)')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                        help='learning rate (default: 0.0001)')
    parser.add_argument('--b1', type=float, default=0.9, metavar='B1',
                        help='beta 1 (default: 0.9)')
    parser.add_argument('--b2', type=float, default=0.98, metavar='B2',
                        help='beta 2 (default: 0.98)')
    parser.add_argument('--no-cuda', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=0, metavar='S',
                        help='random seed (default: 0)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    parser.add_argument('--weight_decay', type=float, default=0, help='Weight decay of Adam Optimizer')
    parser.add_argument('--lm', type=float, default=1.0, help='learning rate multiplier for train c (defualt 0.01)')
    parser.add_argument('--win_size', type=int, default=5)
    parser.add_argument('--max_days', type=int, default=None)
    parser.add_argument('--num_stds', type=float, default=0.2)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--percentile', type=float, default=0.2)
    parser.add_argument('--train_hyp', default=True)
    parser.add_argument('--train_net', default=False)

    args = parser.parse_args()

    squads = os.listdir(args.data_folder)
    if '._.DS_Store' in squads:
        squads.remove('._.DS_Store')

    if '.DS_Store' in squads:
        squads.remove('.DS_Store')

    nada = [0.0]*len(squads)

    parameters_squads = pd.DataFrame({'Squads': squads, 'win_size': [args.win_size]*len(squads), 'num_stds': nada, 'threshold': nada, 'percentile': nada, 't': nada})
    index = list(parameters_squads.index.to_list())
    
    num_stds = [0.0]*len(squads)
    threshold = [0.0]*len(squads)
    percentile = [0.0]*len(squads)
    t = [0.0]*len(squads)

    parameters_squads = pd.read_csv('parameters.csv')
    num_stds = parameters_squads.num_stds.to_list()
    threshold = parameters_squads.threshold.to_list()
    percentile = parameters_squads.percentile.to_list()
    t = parameters_squads.t.to_list()
    squads_no_realizados = ['VERDULEROS', 'HORA DEL TE', 'PLATOS PREPARADOS', 'ABSTEMIOS IMPULSIVOS', 'TUTIFRUTI']

    count_squad = 0
    for squad in squads:

        if squad not in squads_no_realizados:
            count_squad += 1
            continue
        print('-----------------Squad: ', squad, '-----------') 
        squad_folder = os.path.join(args.data_folder, squad)

        trainset = os.path.join(squad_folder, 'train')
        trainset_folder = os.path.join(trainset, 'inputs')
        train_labels_path = os.path.join(trainset, 'labels')

        valset = os.path.join(squad_folder, 'val')
        valset_folder = os.path.join(valset, 'inputs')
        val_labels_path = os.path.join(valset, 'labels')

        #create dataloader
        dataset_train = Dataset(trainset_folder, train_labels_path, args.win_size, max_days=args.max_days)
        dataloader_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, collate_fn=generate_batch)
        print('valsetfolder: ', valset_folder)
        dataset_val = Dataset(valset_folder, val_labels_path, args.win_size, max_days=args.max_days)
        dataloader_val = DataLoader(dataset_val, batch_size=args.batch_size, shuffle=True, collate_fn=generate_batch)

        # make files
        experiment_path = os.path.join(args.root, squad)
        tb_path = os.path.join(experiment_path, 'tensorboard')
        model_path = os.path.join(experiment_path, 'models')

        if not os.path.exists(experiment_path):
            os.makedirs(experiment_path)
        '''
        else:
            print('ERROR! The experiment name already exist.')
            exit()
        '''

        if not os.path.exists(model_path):
            os.makedirs(model_path)

        # config logger
        logging.basicConfig(filename=os.path.join(experiment_path, 'info.log'),
                            filemode='w',
                            format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                            datefmt='%H:%M:%S',
                            level=logging.DEBUG)
        # log argparse
        for arg, value in sorted(vars(args).items()):
            logging.info("Argument %s: %r", arg, value)

        # create dir tensorboard
        writer = SummaryWriter(tb_path)

        writer.add_text('args', str(args), 0)
        writer.add_text('Random Seed: ', str(args.seed), 0)
        #comentar
        cudnn.benchmark = True

        # flag of use cuda device
        #comentar
        use_cuda = not args.no_cuda and torch.cuda.is_available()
        logging.info('use_cuda:' + str(use_cuda))

        # set seed
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

        # set device
        #comentar
        device = torch.device("cuda" if use_cuda else "cpu")
        device = torch.device("cuda" + ":" + str(args.device) if use_cuda else "cpu")
        torch.cuda.set_device(torch.device(device))

        #llamar  red
        predictor = PromoDetecter.PromoDetecter(num_stds=args.num_stds,
            threshold=args.threshold, percentile=args.percentile, device=device, lm=args.lm,
            train_hyp=args.train_hyp, train_net=args.train_net, max_days=args.max_days)

        logging.info(predictor)
        print(predictor)
        
        # pass to double
        predictor.double()
        #hacer uniforme la distribucion por defecto de la red
        for p in predictor.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        predictor = predictor.to(device)

        optimizer = torch.optim.Adam(predictor.parameters(), lr=args.lr,
                                     betas=(args.b1, args.b2), eps=1e-9, weight_decay=args.weight_decay)

        # Loss
        loss_fn = torch.nn.BCEWithLogitsLoss()
        
        #------------------------TRAIN
        logging.info('Start to Train')
        
        print('Start to Train')
        count_iter_train = 0
        count_iter_val = 0

        error_fmax_min = 100

        for epoch in range(args.epochs):
            #train
            print('-------------Train Epoch ', epoch + 1, '-------------')
            train_epoch_loss, count_iter_train = train_epoch(dataloader=dataloader_train,
                                                             epoch=epoch,
                                                             epochs_number=args.epochs,
                                                             count_iter=count_iter_train,
                                                             predictor_net=predictor,
                                                             optimizer=optimizer,
                                                             loss_fn=loss_fn,
                                                             writer=writer,
                                                             device=device)
            writer.add_scalar('Loss/Training_Epoch_Loss', train_epoch_loss, epoch + 1)
            print('Average Train Loss of Epoch ', epoch + 1, '= ', train_epoch_loss)
            #validation
            if (epoch + 1) % args.validation_freq == 0:
                print('******************************VALIDATION******************************')
                val_epoch_loss, count_iter_val, fmax, tmax, error_fmax, fake_negative_emin, t_emin, error_min, fake_negative_fmax = val_epoch(
                    dataloader=dataloader_val,
                    count_iter=count_iter_val,
                    predictor_net=predictor,
                    writer=writer,
                    device=device,
                    loss_fn=loss_fn) 
                
                epoch_val = 1 + int(epoch / args.validation_freq)
                writer.add_scalar('Loss/Validation_Epoch_Loss', val_epoch_loss, epoch_val)
                writer.add_scalar('Score1/F_max', fmax, epoch_val)
                writer.add_scalar('Score1/T_max', tmax, epoch_val)
                writer.add_scalar('Score1/Error_Fmax', error_fmax, epoch_val)
                writer.add_scalar('Score1/fake_negative_fmax', fake_negative_fmax, epoch_val)


                writer.add_scalar('Score2/FN_emin', fake_negative_emin, epoch_val)
                writer.add_scalar('Score2/T_emin', t_emin, epoch_val)
                writer.add_scalar('Score2/error_min', error_min, epoch_val)
                
                if error_fmax <= error_fmax_min:
                    stds, thres, per = predictor.get_real_values()
                    t_value = tmax
                    error_fmax_min = error_fmax

                print('Average Validation Loss of Epoch ', epoch + 1, '= ', val_epoch_loss)
                print('In the epoch ', epoch + 1, ' there is a F_max=', round(fmax, 3), ' with a t_max=', round(tmax,3))

            ## save model
            if (args.save_model) and ((epoch + 1) % args.epochs_save == 0):
                print('++++++++++++++++++++++++++++++SAVING MODEL++++++++++++++++++++++++++++++')
                logging.info('save model')
                torch.save(predictor.state_dict(), model_path + "/model_" + squad + '_epoch_' + str(epoch) + ".pt")
        #cache
        torch.cuda.empty_cache()

        num_stds[count_squad] = stds
        threshold[count_squad] = thres
        percentile[count_squad] = per
        t[count_squad] = t_value
        print('-----Finished Squad-----')
        parameters_squads.num_stds = pd.Series(num_stds, index=index).values
        parameters_squads.threshold = pd.Series(threshold, index=index).values
        parameters_squads.percentile = pd.Series(percentile, index=index).values
        parameters_squads.t = pd.Series(t, index=index).values
        parameters_squads.to_csv('parameters.csv', index=False)
        print('Parameters for squad ', squad, ' saved.')
        count_squad += 1