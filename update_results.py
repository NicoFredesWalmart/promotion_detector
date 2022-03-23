from classify_data import *

			 	

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