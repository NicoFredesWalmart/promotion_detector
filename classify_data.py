import pandas as pd
import os
import numpy as np
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from loaddata import Dataset, generate_batch
from pathlib import Path
import PromoDetecter
import numpy as np
import sys
import json
import pickle
from datetime import datetime
import shutil

def predict(x, x0, l, l0, pad, pad0, predictor_net, device):
    x = x.double().to(device)
    x0 = x0.double().to(device)
    l = l.double().to(device)
    l0 = l0.double().to(device)
    pad = pad.double().to(device)
    pad0 = pad0.double().to(device)

    # usar la red
    prediction = predictor_net(x, x0, l, l0, pad, pad0).detach()

    # arreglar las dimensiones cuando solo se carga una proteina
    if prediction.shape[0] == 1:
        labels = labels.unsqueeze(0)

    return prediction


def inference(dataloader, predictor_net, device, t):
    #valor para la funcion escalon cuando la entrada es 0
  
    values = torch.tensor([1.0]).double().to(device)
    barcodes_all = []
    l0_all = []

    for i, (x, x0, l, l0, pad, pad0, barcodes) in enumerate(dataloader):

        prediction = predict(x, x0, l, l0, pad, pad0, 
        	predictor_net=predictor_net, device=device)
		#sigmoide
        prediction = torch.sigmoid(prediction)
        #------calculos del data en general
        prediction_zeros = torch.zeros(prediction.shape[0], 100)

        prediction_zeros[:prediction.shape[0], :prediction.shape[1]] = prediction

        l0 = torch.tensor(l0).reshape(prediction.shape[0])
        l0 = l0.tolist()
        barcodes_all = barcodes_all + barcodes
        l0_all = l0_all + l0
        if i==0:
            predictions_all = prediction_zeros
        else:
            predictions_all = torch.cat((predictions_all, prediction_zeros), 0)
    
    predictions_all = predictions_all.to(device)
    
    t = torch.tensor([t]).double().to(device)
    round_prediction = torch.heaviside((predictions_all - t).type(torch.float64), values)
    round_prediction = torch.ones(round_prediction.shape) - round_prediction
    round_prediction = round_prediction.type(torch.int64)

    return round_prediction, l0_all, barcodes_all


def data_test_adapter(data_path):
	
	cat_folders = os.listdir(data_path)

	count_n = 0
	count_u = 0
	for folder in cat_folders:
		nielsen_path = os.path.join(data_path, folder, 'prices_nielsen_classification.csv')
		utt_path = os.path.join(data_path, folder, 'prices_utt_classification.csv')

		if os.path.exists(nielsen_path):
			count_n += 1
			nielsen = pd.read_csv(nielsen_path)
			if count_n == 1:
				all_nielsen = nielsen
			else:
				all_nielsen = pd.concat([all_nielsen, nielsen], ignore_index=True)

		if os.path.exists(utt_path):
			count_u += 1
			utt = pd.read_csv(utt_path)
			if count_u == 1:
				all_utt = utt
			else:
				all_utt = pd.concat([all_utt, utt], ignore_index=True)

	data_path = os.path.join(data_path, 'todo')
	
	if not os.path.exists(data_path):
		os.mkdir(data_path)

	all_nielsen_path = os.path.join(data_path, 'all_nielsen.csv')	
	all_utt_path = os.path.join(data_path, 'all_utt.csv')	
	
	data_list = [(all_nielsen, all_nielsen_path), (all_utt, all_utt_path)]

	cat2squad = pd.read_csv('notebooks/code_cat_squad.csv')
	cat2squad = cat2squad[cat2squad['Cod_Categoria']!=' ']
	cat2squad = cat2squad.astype({'Cod_Categoria':np.int64})
	cat2squad = cat2squad.set_index('Cod_Categoria')
	cod2squad = cat2squad.to_dict('index')
	
	for (data, data_path) in data_list:
		data = data.rename(columns={"registered_at": "date", 'competitor_store_nbr': 'Tienda_Number'})
		data['Cod_Categoria'] = data.apply(lambda row : int(row['category'][:5]), axis = 1)
		data = data.astype({'Cod_Categoria':np.int64, 'barcode':np.int64, 'price':np.int64})
		data = data[data.Cod_Categoria.isin(list(cod2squad.keys()))]
		data['Categoria'] = data.apply(lambda row : cod2squad[row['Cod_Categoria']]['Categoria'], axis = 1)
		data['Squad'] = data.apply(lambda row : cod2squad[row['Cod_Categoria']]['Squad'], axis = 1)
		data.to_csv(data_path, index=False)

	df_nielsen = pd.read_csv(all_nielsen_path)
	df_utt = pd.read_csv(all_utt_path)

	return df_nielsen, df_utt


def data_adapter(data):
	
	data = data.rename(columns={"MasterPrice_Dt": "date", "Barcode": "barcode", 'Precio':'price'})
	data = data.astype({'Cod_Categoria':np.int64, 'barcode':np.int64, 'price':np.int64})

	cat2squad = pd.read_csv('code_cat_squad.csv')
	cat2squad = cat2squad[cat2squad['Cod_Categoria']!=' ']
	cat2squad = cat2squad.astype({'Cod_Categoria':np.int64})
	cat2squad = cat2squad.set_index('Cod_Categoria')
	cod2squad = cat2squad.to_dict('index')
	data = data[data.Cod_Categoria.isin(list(cod2squad.keys()))]
	data['Categoria'] = data.apply(lambda row : cod2squad[row['Cod_Categoria']]['Categoria'], axis = 1)
	data['Squad'] = data.apply(lambda row : cod2squad[row['Cod_Categoria']]['Squad'], axis = 1)
	return data

def test(data_path, squad, device, t, win_size, batch_size,
	num_stds, threshold, percentile, classified_data_path, seed=0):

	torch.manual_seed(seed)
	fecha = datetime.now()

	dfs = data_test_adapter(data_path)

	predictor = PromoDetecter.PromoDetecter(num_stds=num_stds,
		threshold=threshold, percentile=percentile, device=device, train_hyp=False)
	predictor.double()
	#hacer uniforme la distribucion por defecto de la red
	for p in predictor.parameters():
		if p.dim() > 1:
			nn.init.xavier_uniform_(p)

	predictor = predictor.to(device)

	fuente = 'nielsen'
	for data in dfs:

		input_path, output_path, date = tensor_generator(data, squad, fecha, fuente)

		dataset = Dataset(input_path, win_size)
		dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
			collate_fn=generate_batch)

		round_prediction, l0_all, codes_all = inference(dataloader=dataloader,
			predictor_net=predictor,
			device=device, t=t)

		round_prediction = torch.ones(round_prediction.shape) - round_prediction

		for fila in range(len(codes_all)):
			code = codes_all[fila]
			n_prices = l0_all[fila]
			predictions = round_prediction[fila, :n_prices].reshape(n_prices)
			predictions = predictions.tolist()
			final_path = os.path.join(output_path, str(code) + '.csv')
			sample_path = os.path.join(input_path, str(code) + '.pt')
			os.remove(sample_path)
			output = pd.read_csv(final_path)
			count = 0
			output['is_regular_price'] = predictions

			output.to_csv(final_path, index=False)

			if fila==0:
				df = output
			else:
				df = pd.concat([df, output], ignore_index=True)

		classified_data_path_2 = os.path.join(classified_data_path, squad)
		
		if not os.path.exists(classified_data_path_2):
			os.mkdir(classified_data_path_2)

		classified_data_path_2 = os.path.join(classified_data_path_2, fuente + '_' + date + '.csv')

		df.to_csv(classified_data_path_2, index=False)
		print('------- ', fuente, ' results-----')
		accuracy = (df.true_classification==df.is_regular_price).sum()/len(df)
		fp = ((df.true_classification!=df.is_regular_price) & (df.is_regular_price==1)).sum()/len(df)
		fn = ((df.true_classification!=df.is_regular_price) & (df.is_regular_price==0)).sum()/len(df)		
		fp_p = ((df.true_classification!=df.is_regular_price) & (df.is_regular_price==1)).sum()/(1e-12 + (df.true_classification==0).sum())
		print('Accuracy: ', round(accuracy*100, 2), '%')
		print('Fake Negative: ', round(fn*100, 2), '%')
		print('Fake Positive (IMPORTANTE): ', round(fp*100, 2), '%')
		print('Fake Positive / P: ', round(fp_p*100, 2), '%')
		fuente = 'utt'

def tensor_generator(data, fuente, squads):

	data['date'] = pd.to_datetime(data['date'])
	data = data.astype({'barcode': str, 'Tienda_Number': str})
	#data['BarStoreCode'] = data.apply(lambda row : row['barcode'] + '_' + row['Tienda_Number'], axis = 1)
	data['BarStoreCode'] = data['barcode'] + data['Tienda_Number']

	codes = data.BarStoreCode.to_list()
	input_folder = os.path.join('..', 'inputs')
	output_folder = os.path.join('..', 'outputs')

	if not os.path.exists(input_folder):
		os.mkdir(input_folder)

	if not os.path.exists(output_folder):
		os.mkdir(output_folder)

	inputs = os.path.join(input_folder, fuente)
	outputs = os.path.join(output_folder, fuente)

	if os.path.exists(inputs):
		shutil.rmtree(inputs, ignore_errors=True)

	if os.path.exists(outputs):
		shutil.rmtree(outputs, ignore_errors=True)

	os.mkdir(inputs)
	os.mkdir(outputs)

	for squad in squads:
		os.mkdir(os.path.join(inputs, squad))
		os.mkdir(os.path.join(outputs, squad))

	for code in codes:
	    arrow_data = data[data.BarStoreCode == code].sort_values(by='date',ascending=False)
	    new_squad = list(arrow_data.Squad.to_list())[0]
	    if new_squad in squads:
	    	SQUAD = new_squad
	    else:
	    	SQUAD = 'OTRO'
	    final_path = os.path.join(outputs, SQUAD, str(code) + '.csv')
	    arrow_data.to_csv(final_path, index=False)
	    sample = torch.tensor(arrow_data.price.to_list())
	    sample_path = os.path.join(inputs, SQUAD, str(code) + '.pt')
	    torch.save(sample, sample_path)

	return inputs, outputs

def output_generator(dataloader, predictor_net, device, t, squad):

	values = torch.tensor([1.0]).double().to(device)
	t = torch.tensor([t]).double().to(device)

	for i, (x, x0, l, l0, pad, pad0, barcodes) in enumerate(dataloader):

		print('++++++', squad, ': Inferring Batch ', i, '/', len(dataloader))

		prediction = predict(x, x0, l, l0, pad, pad0, predictor_net=predictor_net, device=device)
		#sigmoide
		prediction = torch.sigmoid(prediction)

		prediction = torch.heaviside((prediction - t).type(torch.float64), values)
		prediction = torch.ones(prediction.shape) - prediction
		prediction = prediction.type(torch.int64)

		l0 = torch.tensor(l0).reshape(prediction.shape[0])
		barcodes = torch.tensor(barcodes).reshape(prediction.shape[0])
		l0 = l0.tolist()
		barcodes = barcodes.tolist()

		for fila in range(len(barcodes)):
			n_prices = l0[fila]
			code = barcodes[fila]
			one_prediction = prediction[fila, :n_prices].reshape(n_prices)
			one_prediction = one_prediction.tolist()
			final_path = os.path.join(output_path, str(code) + '.csv')
			sample_path = os.path.join(input_path, str(code) + '.pt')
			os.remove(sample_path)
			output = pd.read_csv(final_path)
			output['is_regular_price'] = predictions

			output.to_csv(final_path, index=False)

			if i==0 and fila==0:
				df = output
			else:
				df = pd.concat([df, output], ignore_index=True)

	return df

def main(input_data, model_parameters, device, batch_size, classified_data_path, fuente):

	print('Starting')

	torch.manual_seed(0)
	fecha = datetime.now()

	print('Adapting Data...')
	
	data = data_adapter(input_data)

	print('Generating tensors...')

	squads = model_parameters.Squads.to_list()

	inputs, outputs = tensor_generator(data, fuente, squads)
	squad_counter = 0

	print('Starting Squads Loop...')

	for squad in squads:

		print('Starting Squad ', squad, '...')

		input_path = os.path.join(inputs, squad)
		output_path = os.path.join(outputs, squad)

		squad_parameters = model_parameters[model_parameters.Squads==squad]
		num_stds = squad_parameters.num_stds[0]
		threshold = squad_parameters.threshold[0]
		percentile = squad_parameters.percentile[0]
		win_size = squad_parameters.win_size[0]
		t = squad_parameters.t[0]
		
		predictor = PromoDetecter.PromoDetecter(num_stds=num_stds,
			threshold=threshold, percentile=percentile, device=device, train_hyp=False)

		predictor.double()

		predictor = predictor.to(device)

		dataset = Dataset(input_path, win_size)
		dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
			collate_fn=generate_batch)

		print('------', squad, ': Output Generator...')

		df = output_generator(dataloader=dataloader,
			predictor_net=predictor,
			device=device, t=t, squad=squad)

		if squad_counter==0:
			df_fuente = df
		else:
			df_fuente = pd.concat([df_fuente, df], ignore_index=True)

		squad_counter += 1
			
	print('********** Saving Output DataFrames *********')
	df_fuente.to_csv(os.path.join(outputs, 'output.csv'), index=False)

	output_path = os.path.join(classified_data_path, fuente + '.csv')
	df_fuente.to_csv(output_path, index=False)
	print('********** Process Finished *********')
			 	

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("-data2classify", default='../data2classify/nielsen.csv')
	parser.add_argument("-classified_data_path", default='../classified_data/')
	parser.add_argument('-batch_size', default=64, type=int, help='batch size')
	parser.add_argument('-device', type=int, default=0)
	parser.add_argument("-parameters_path", default='parameters.csv')
	parser.add_argument('--no-cuda', default=True, help='disables CUDA')

    			
	args = parser.parse_args()

	use_cuda = not args.no_cuda and torch.cuda.is_available()

	device = torch.device("cuda" + ":" + str(args.device) if use_cuda else "cpu")

	parameters = pd.read_csv(args.parameters_path)

	data2classify = pd.read_csv(args.data2classify)
   
	main(input_data=data2classify,
		model_parameters=parameters,
		device=device,
		batch_size=args.batch_size,
    	classified_data_path=args.classified_data_path,
    	fuente='nielsen')