# LSTM for predicting winners with pick and vegas - normal algorithm
import numpy
import matplotlib.pyplot as plt
import pandas
import math
import time
import copy
from IPython.display import SVG
from keras.models import Sequential
from keras.layers import Activation, Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.callbacks import EarlyStopping
from keras.layers.embeddings import Embedding
from keras.utils.visualize_util import plot
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from IPython.display import SVG
from keras.utils.visualize_util import model_to_dot
from datetime import datetime

# convert an array of values into a dataset matrix
def normalize_team(dataset):
	team_matrix = []
	team_names = []
	for i in range(0, len(dataset)):
		time = dataset[i, 3]
		team = mapping(dataset[i, 4])
		opponent_team = mapping(dataset[i, 5])
		teamname = dataset[i, 4]
		opponteamname = dataset[i, 5]
		point = dataset[i, 6]
		opponent_point = dataset[i, 7]
		spread = dataset[i, 9]
		point_diff = numpy.subtract(numpy.int_(point), numpy.int_(opponent_point))
		diff = numpy.add(point_diff, spread)
		consensus_sp = 1
		if dataset[i, 11] >= 50:
			if diff > 0:
				consensus_sp = 1
			else:
				consensus_sp = 0
		else:
			if diff > 0:
				consensus_sp = 0
			else:
				consensus_sp = 1
		opponent_pick = 0
		if i < len(dataset) - 1:
			opponent_pick = dataset[i + 1, 10]
		else:
			opponent_pick = 0
		team_matrix.append(numpy.concatenate((team, opponent_team, [time.split(':', 1)[0], opponent_pick])))
		team_names.append([teamname, opponteamname, consensus_sp, spread, point, opponent_point])
	dataset = numpy.concatenate((dataset, team_matrix), axis=1)
	dataset = numpy.delete(dataset, [1, 3, 4, 5], axis = 1)
	dataset = dataset[::2]
	team_names = team_names[::2]
	return numpy.array(dataset), numpy.array(team_names)

# convert an array of values into a dataset matrix
def create_dataset(data):
	dataX, dataY, team_matrix = [], [], []
	dataX = numpy.delete(data, [1, 2, 3], axis = 1)
	for i in range(0, len(data)):
		dataY.append([data[i, 3]])
	return numpy.array(dataX), numpy.array(dataY)

# standardize name of teams
def mapping(name):
	# training dataset generation
	teams = [ 'DAL', 'WAS', 'POR', 'NYK', 'SAC', 'DET', 'ATL', 'NOP', 'LAC', 'DEN', 'PHI', 'LAL', 'CLE', 'CHI', 'IND', 'TOR', 'ORL', 'MIA', 'MEM', 'BOS', 'MIL', 'BKN', 'OKC', 'HOU', 'PHX', 'MIN', 'UTA', 'GSW', 'SAS', 'CHA', 'EAST', 'WEST', 'WALLSTAR', 'USALLSTAR' ]
	normalized_values = numpy.eye(34)
	index = teams.index(name)
	return normalized_values[index]

# get Percentage of weekly prediction
def getPercentage(data):
	percent1 = 0
	percent2 = 0
	percent3 = 0
	result_winloss_array = []
	result_winloss_array2 = []
	data = numpy.array(data)
	for i in range(0, len(data)):
		point_diff1 = numpy.subtract(numpy.int_(data[i, 4]), numpy.int_(data[i, 5]))
		diff1 = numpy.sign(numpy.add(point_diff1, numpy.float_(data[i, 3])))
		point_diff2 = numpy.int_(data[i, 6])
		diff2 = numpy.sign(numpy.add(point_diff2, numpy.float_(data[i, 3])))

		if diff1 == diff2:
			percent1 = percent1 + 1
			result_winloss_array.append(1)
		else:
			result_winloss_array.append(0)

		if numpy.sign(point_diff1) == numpy.sign(point_diff2):
			percent3 = percent3 + 1
			result_winloss_array2.append(1)
		else:
			result_winloss_array2.append(0)

		if numpy.int_(data[i, 2]) == 1:
			percent2 = percent2 + 1
	value1 = (percent1 / float(len(data))) * 100
	value2 = (percent2 / float(len(data))) * 100
	value3 = (percent3 / float(len(data))) * 100
	return numpy.around(value1, decimals = 2), numpy.around(value2, decimals = 2), numpy.around(value3, decimals = 2), result_winloss_array, result_winloss_array2

# get lengths of both training and testing data from csv
def getLength(dataset, date):
	num_date = numpy.searchsorted(dataset[:,1], date)
	trainLength = num_date - 1
	num_date = [i for i,val in enumerate(dataset[:,1]) if val == date]
	testLength = num_date[-1]
	return trainLength, testLength

def signWithMargin(x):
	if x > 0.1:
		return 1
	elif x < -0.1:
		return -1
	else:
		return 0

# get the good chance from line movement
def getChancefromLineMovement(item, data_spread):

	datetimeFormat = '%m%d%H:%M%p'
	item[3] = item[3][:-1] + 'PM'
	str_startDate = str(item[2])[4:] + item[3]
	print ('-----------str_startDate-----------', str_startDate)
	startTime = datetime.strptime(str_startDate, datetimeFormat)
	
	data_pinnacle = []
	data_bookmaker = []
	data_sports_interaction = []
	data_bovada = []

	for i in range(0, len(data_spread)):
		if item[2] == data_spread[i, 1] and item[4] == data_spread[i, 3] and item[5] == data_spread[i, 4]:
			if data_spread[i, 0] == 'pinnacle':
				data_pinnacle.append(data_spread[i])
			elif data_spread[i, 0] == 'bookmaker':
				data_bookmaker.append(data_spread[i])
			elif data_spread[i, 0] == 'sports interaction':
				data_sports_interaction.append(data_spread[i])
			else:
				data_bovada.append(data_spread[i])

	if not data_pinnacle:
		return 0

	else:
		data_pinnacle = numpy.array(data_pinnacle)
		data_bookmaker = numpy.array(data_bookmaker)
		data_sports_interaction = numpy.array(data_sports_interaction)
		data_bovada = numpy.array(data_bovada)

		time_array_pin = []
		spread_array_pin = []
		time_array_book = []
		spread_array_book = []
		time_array_sports = []
		spread_array_sports = []
		time_array_bovada = []
		spread_array_bovada = []

		def getArray(data):
			time_array = []
			spread_array = []
			for i in range(0, len(data)):
				datetimeFormat = '%m/%d %H:%M %p'
				tdelta = datetime.strptime (data[0, 5], datetimeFormat) - datetime.strptime (data[i, 5], datetimeFormat)
				tdelta = int(tdelta.total_seconds () / 60)
				spread = data[i, 6].split ( )[0]
				if spread != 'PK':
					time_array.append (tdelta)
					spread_array.append (float(spread))
				else:
					time_array.append (tdelta)
					spread_array.append (float(0))
			return time_array, spread_array

		time_array_pin, spread_array_pin = getArray(data_pinnacle)
		time_array_book, spread_array_book = getArray(data_bookmaker)
		time_array_sports, spread_array_sports = getArray(data_sports_interaction)
		time_array_bovada, spread_array_bovada = getArray(data_bovada)

		# print ('\ntime_array_book =', time_array_book)
		# print ('\nspread_array_book =', spread_array_book)
		# print ('\ntime_array_pin =', time_array_pin)
		# print ('\nspread_array_pin_real =', spread_array_pin)
		# print ('\nspread_array_pin_inter_book =', numpy.interp(time_array_pin, time_array_book, spread_array_book))
		# print ('\nspread_array_pin_inter_sports =', numpy.interp(time_array_pin, time_array_sports, spread_array_sports))
		# print ('\nspread_array_pin_inter_bovada =', numpy.interp(time_array_pin, time_array_bovada, spread_array_bovada))

		# plt.plot(time_array_pin, spread_array_pin, 'o')
		# plt.plot(time_array_pin, numpy.interp(time_array_pin, time_array_book, spread_array_book), '-x')
		# plt.plot(time_array_pin, numpy.interp(time_array_pin, time_array_sports, spread_array_sports), '--')
		# plt.plot(time_array_pin, numpy.interp(time_array_pin, time_array_bovada, spread_array_bovada), 'b')
		# plt.show()

		if not spread_array_book:
			time_array_book = time_array_pin;
			spread_array_book = spread_array_pin;
		if not time_array_sports:
			time_array_sports = time_array_pin;
			spread_array_sports = spread_array_pin;
		if not spread_array_bovada:
			time_array_bovada = time_array_pin;
			spread_array_bovada = spread_array_pin;

		print ('time_array_pin', time_array_pin)

		spread_array_pin_inter_book = numpy.interp(time_array_pin, time_array_book, spread_array_book)
		spread_array_pin_inter_sports = numpy.interp(time_array_pin, time_array_sports, spread_array_sports)
		spread_array_pin_inter_bovada = numpy.interp(time_array_pin, time_array_bovada, spread_array_bovada)

		result = 0.0
		for i in range(len(spread_array_pin) - 1, 1, -1):
			del_pin = spread_array_pin[i] - spread_array_pin[i - 1]
			del_book = spread_array_pin_inter_book[i] - spread_array_pin_inter_book[i - 1]
			del_sports = spread_array_pin_inter_sports[i] - spread_array_pin_inter_sports[i - 1]
			del_bovada = spread_array_pin_inter_bovada[i] - spread_array_pin_inter_bovada[i - 1]

			sign_del_pin = signWithMargin(del_pin)
			sign_del_book = signWithMargin(del_book)
			sign_del_sports = signWithMargin(del_sports)
			sign_del_bovada = signWithMargin(del_bovada)

			print ('del_pin', spread_array_pin[i] - spread_array_pin[i - 1])
			print ('del_book', spread_array_pin_inter_book[i] - spread_array_pin_inter_book[i - 1])
			print ('del_sports', spread_array_pin_inter_sports[i] - spread_array_pin_inter_sports[i - 1])
			print ('del_bovada', spread_array_pin_inter_bovada[i] - spread_array_pin_inter_bovada[i - 1])
			print ('------------------')

			print ('sign_del_pin', sign_del_pin)
			print ('sign_del_book', sign_del_book)
			print ('sign_del_sports', sign_del_sports)
			print ('sign_del_bovada', sign_del_bovada)
			print ('------------------')

			if del_pin != 0.0:
				if sign_del_pin == sign_del_book:
					if sign_del_pin == sign_del_sports:
						if sign_del_pin == sign_del_bovada:
							result = result
						else:
							result = result + sign_del_pin
					elif sign_del_pin == sign_del_bovada:
						result = result + sign_del_pin
					else:
						result = result + 2 * sign_del_pin
				elif sign_del_pin == sign_del_sports:
					if sign_del_pin == sign_del_bovada:
						result = result + sign_del_pin
					else:
						result = result + 2 * sign_del_pin
				else:
					if sign_del_pin == sign_del_bovada:
						result = result + 2 * sign_del_pin
					else:
						result = result + 3 * sign_del_pin
		print ("------result------", result)
		return result

# fix random seed for reproducibility
numpy.random.seed(7)

# load the dataset
dataframe = pandas.read_csv('seeds/inputs.csv', usecols = [1, 2, 3, 4, 7, 8, 9, 10, 11, 12, 14, 19, 20, 21], engine = 'python')
dataset = dataframe.values

dataframe_spread = pandas.read_csv ('seeds/nba_timelines.csv', usecols = [0, 1, 2, 5, 6, 7, 8, 9], engine = 'python')
dataset_spread = dataframe_spread.values

print ('dataset', dataset[3])

print ('dataset_spread', dataset_spread[3])

print ('Calculating Line Movement history...')

lineMovementParams = []
lineMovementParams = [0.0] * len(dataset)
for i in range(len(dataset)):
	predicted_spread = getChancefromLineMovement(dataset[i], dataset_spread)
	if predicted_spread != 0:
		lineMovementParams[i] = predicted_spread

print ('dataset[3]', dataset[3])
print ('dataset[4]', dataset[4])
print ('lineMovementParams', lineMovementParams)
# lineMovementParams = normalizeLineMovements(lineMovementParams)

# dataset, teamnames = normalize_team(dataset)
# dataset_clone = dataset

# # print ('------dataset[3]------', dataset[3])

# # get maximum of point difference
# point_diff_max = numpy.amax(dataset[:, 4])
# point_diff_min = numpy.amin(dataset[:, 4])

# # print ('------point_diff_max------', point_diff_max)
# # print ('------point_diff_min------', point_diff_min)

# # get maximum of spread
# spread_max = numpy.amax(dataset[:, 5])
# spread_min = numpy.amin(dataset[:, 5])

# # print ('------spread_max------', spread_max)
# # print ('------spread_min------', spread_min)

# dataset = dataset.astype('float32')

# # normalize the dataset
# scaler = MinMaxScaler(feature_range = (0, 1))
# dataset = scaler.fit_transform(dataset)

# dataset = numpy.delete(dataset, [1], axis = 1)

# # print ('dataset[3]', dataset[3])

# # create and fit the LSTM network
# dim_inputs = 75
# dim_outputs = 1

# mem_blocks = 2
# tt_batch_size = 1
# tp_batch_size = 1

# select_date = input ("Please type the date for which you would like to get (for example: 20160210):")
# select_date = int (select_date)

# model = Sequential()
# model.add(LSTM(mem_blocks, input_dim = dim_inputs))
# model.add(Dropout(0.2))
# model.add(Dense(dim_outputs, activation = 'tanh'))
# model.compile(loss = 'mean_squared_error', optimizer = 'adam', metrics = ['accuracy'])

# model = Sequential()
# model.add(LSTM(input_dim = dim_inputs, output_dim = 300, return_sequences = True))  
# model.add(LSTM(input_dim = 300, output_dim = 500, return_sequences = True))  
# model.add(Dropout(0.2))
# model.add(LSTM(input_dim = 500, output_dim = 200, return_sequences = False))  
# model.add(Dropout(0.2))
# model.add(Dense(input_dim = 200, output_dim = dim_outputs))  
# model.add(Activation('sigmoid'))
# model.compile(loss = 'mean_squared_error', optimizer = 'adam', metrics = ['accuracy'])

# # split into train and test sets
# trainLength, testLength = getLength(dataset_clone, select_date)

# train, test = dataset[0:trainLength,:], dataset[trainLength:testLength,:]
# trainX, trainY = create_dataset(train)
# testX, testY = create_dataset(test)

# testTeams = teamnames[trainLength:testLength,:]

# # reshape input to be [samples, time steps, features]
# trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
# testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

# tt_nb_epoch = 200

# # training NN Model
# early_stopping = EarlyStopping(monitor = 'val_loss', min_delta = 0.0000001, patience = 2, verbose = 0, mode = 'max')
# hist = model.fit(
# 	trainX,
# 	trainY,
# 	nb_epoch = tt_nb_epoch,
# 	batch_size = tt_batch_size,
# 	shuffle = True,
# 	show_accuracy = True,
# 	verbose = 1,
# 	validation_split = 0.1,
# 	callbacks = [early_stopping]
# )
# print hist.history
# # print (hist.history)

# # evaluate the network
# loss, accuracy = model.evaluate(trainX, trainY)
# # print "\nLoss: %.2f, Accuracy: %.2f%%" % (loss, accuracy * 100)
# # print ("\nLoss: %.2f, Accuracy: %.2f%%" % (loss, accuracy * 100))

# # make predictions
# testPredict = model.predict(testX, batch_size = tp_batch_size, verbose = 1)

# testPredict = testPredict * (point_diff_max - point_diff_min) + point_diff_min

# result = [[int(numpy.around(x, decimals = 0))] for x in testPredict]
# testPredict = numpy.concatenate((numpy.array(testTeams), result), axis = 1)

# NN_Percent, Consensus_Percent, WinLoss_Percent, winloss_array, Winner_array = getPercentage(testPredict)

# winloss = [[x] for x in winloss_array]
# modified_testPredict = numpy.concatenate((testPredict, winloss), axis = 1)

# winloss1 = [[x] for x in Winner_array]
# modified_testPredict = numpy.concatenate((modified_testPredict, winloss1), axis = 1)

# # print '\nmodified_testPredict=========', modified_testPredict

# winloss = [[x] for x in winloss_array]
# modified_testPredict = numpy.concatenate((testPredict, winloss), axis = 1)

# print '--------------The prediction Result for date', select_date, '---------------'
# print modified_testPredict
# print 'The percentage of Result for date', select_date, ' was ', NN_Percent, '%.'
# print 'The percentage of Consensus Prediction for date', select_date, ' was ', Consensus_Percent, '%.'

# # print '--------------The prediction Result for week', i, '---------------'
# # print modified_testPredict
# # print 'The percentage of Result for week', i, ' was ', NN_Percent, '%.'
# # print 'The percentage of Consensus Prediction for week', i, ' was ', Consensus_Percent, '%.'

# # # write the prediction result to csv file named result.csv
# # filename = 'results/resultNormalWithVPT-' + str(i) + '-' + time.strftime("%Y%m%d-%H%M%S") + '.csv'
# # with open(filename, "wb") as f:
# # 	numpy.savetxt(f, testPredict, delimiter=",", fmt="%s")

# # # save Model
# # filename = 'saveModels/ModelNormalWithVPT.h5'
# # model.save(filename)
# # print 'The trained Model saved successfully!'
# # # print ('The trained Model saved successfully!')