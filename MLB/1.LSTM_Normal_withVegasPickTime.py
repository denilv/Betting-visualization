# LSTM for predicting winners with pick and vegas - normal algorithm
from __future__ import division
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

teams = []
def sigmoid(c, x):
  return 1 / (1 + math.exp(-c * x))

def american_to_decimal(data):
	if data < 0:
		return (100 / abs(data)) + 1
	else:
		return (abs(data) / 100) + 1

# convert an array of values into a dataset matrix
def features_extract(dataset, teams):	
	team_matrix = []
	team_names = []
	for i in range(0, len(dataset)):
		time = dataset[i, 2]
		team = mapping(dataset[i, 3], teams)
		opponent_team = mapping(dataset[i, 4], teams)
		teamname = dataset[i, 3]
		opponteamname = dataset[i, 4]
		point = dataset[i, 5]
		opponent_point = dataset[i, 6]
		moneyline = dataset[i, 9]
		opponent_moneyline = dataset[i, 10]
		dataset[i, 9] = american_to_decimal(dataset[i, 9])
		dataset[i, 10] = american_to_decimal(dataset[i, 10])

		# if dataset[i, 9] < 0:
		# 	moneyline = (100 / dataset[i, 9]) + 1
		# else:
		# 	moneyline = (dataset[i, 9] / 100) + 1

		# moneyline = dataset[i, 9]
		# opponent_moneyline = dataset[i, 10]

		point_diff = numpy.subtract(numpy.int_(point), numpy.int_(opponent_point))
		consensus_sp = 1
		if dataset[i, 8] >= 50:
			if point_diff > 0:
				consensus_sp = 1
			else:
				consensus_sp = 0
		else:
			if point_diff > 0:
				consensus_sp = 0
			else:
				consensus_sp = 1
		# opponent_pick = 0
		# if i < len(dataset) - 1:
		# 	opponent_pick = dataset[i + 1, 10]
		# else:
		# 	opponent_pick = 0
		team_matrix.append(numpy.concatenate((team, opponent_team, [time.split(':', 1)[0]])))
		team_names.append([time, teamname, opponteamname, consensus_sp, point, opponent_point, moneyline, opponent_moneyline])
	dataset = numpy.concatenate((dataset, team_matrix), axis=1)
	dataset = numpy.delete(dataset, [2, 3, 4], axis = 1)
	dataset = dataset[::2]
	team_names = team_names[::2]

	return numpy.array(dataset), numpy.array(team_names)

def get_confidence(d_set, testPredict, select_date):

	win_loss = []
	d_set = d_set[::2]
	home_margin = 0
	num_date = numpy.searchsorted(d_set[:,1], select_date)
	trainLength = num_date
	num_date = [i for i,val in enumerate(d_set[:,1]) if val == select_date]
	testLength = num_date[-1]

	k = 0
	for i in range(trainLength, testLength + 1):
		team1_pos, team1_nav, team2_pos, team2_nav = 0, 0, 0, 0
		for j in range(trainLength - 500, trainLength):
			if d_set[j][3] == d_set[i][3]:
				diff = numpy.subtract(numpy.int_(d_set[j, 5]), numpy.int_(d_set[j, 6]))
				if diff > 0:
					team1_pos += 1
				else:
					team1_nav += 1
			if d_set[j][3] == d_set[i][4]:
				diff = numpy.subtract(numpy.int_(d_set[j, 5]), numpy.int_(d_set[j, 6]))
				if diff > 0:
					team2_pos += 1
				else:
					team2_nav += 1
		alpha = (team1_pos - team1_nav) - (team2_pos - team2_nav + home_margin)
		# print '-----alpha------', alpha
		# print '-----d_set[i][8]------', d_set[i][8]
		# print '-----testPredict[k][0]------', testPredict[k][0]
		# print '-----x------', d_set[i][8] + testPredict[k][0] - alpha * 0.8
		x = sigmoid(0.1, testPredict[k][0] - alpha * 0.8)
		print (d_set[i, 3], d_set[i, 4], str(team1_pos)+'/'+str(team1_nav), str(team2_pos)+'/'+str(team2_nav))
		win_loss.append(x)
		k += 1
	return numpy.array(win_loss)

# convert an array of values into a dataset matrix
def create_dataset(data):

	dataX, dataY, team_matrix = [], [], []
	dataX = numpy.delete(data, [1, 2, 3], axis = 1)
	for i in range(0, len(data)):
		dataY.append([data[i, 3]])
	return numpy.array(dataX), numpy.array(dataY)

# standardize name of teams
def mapping(name, teams):

	# training dataset generation
	normalized_values = numpy.eye(len(teams))
	index = teams.index(name)

	return normalized_values[index]

# get Percentage of weekly prediction
def get_percentage(data):

	# percent1 = 0
	percent2 = 0
	percent3 = 0
	result_winloss_array = []
	result_winloss_array2 = []
	data = numpy.array(data)

	for i in range(0, len(data)):
		point_diff1 = numpy.subtract(numpy.int_(data[i, 4]), numpy.int_(data[i, 5]))
		point_diff2 = numpy.int_(data[i, 8])

		# if point_diff1 == point_diff2:
		# 	percent1 = percent1 + 1
		# 	result_winloss_array.append(1)
		# else:
		# 	result_winloss_array.append(0)

		if numpy.sign(point_diff1) == numpy.sign(point_diff2):
			percent3 = percent3 + 1
			result_winloss_array.append(1)
		else:
			result_winloss_array.append(0)

		if numpy.int_(data[i, 3]) == 1:
			percent2 = percent2 + 1
	# value1 = (percent1 / float(len(data))) * 100
	value2 = (percent2 / float(len(data))) * 100
	value3 = (percent3 / float(len(data))) * 100

	return numpy.around(value2, decimals = 2), numpy.around(value3, decimals = 2), result_winloss_array

# get lengths of both training and testing data from csv
def get_train_test_timeline_lengths(timeline, date):
	num_date = numpy.searchsorted(timeline, date)
	trainLength = num_date
	# trainLength = num_date - 1
	num_date = [i for i,val in enumerate(timeline) if val == date]
	testLength = num_date[-1] + 1
	# testLength = num_date[-1]

	return trainLength, testLength

def build_train_test_datasets(dataset_clone, dataset, select_date, teamnames):
	trainLength, testLength = get_train_test_timeline_lengths(
		dataset_clone[:, 1],
		select_date
	)

	train, test = dataset[0:trainLength, :], dataset[trainLength:testLength, :]
	trainX, trainY = create_dataset(train)
	testX, testY = create_dataset(test)
	testTeams = teamnames[trainLength:testLength,:]

	trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
	testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

	return trainX, trainY, testX, testY, testTeams

def build_model(input_dim, output_dim):
	model = Sequential()

	model.add(LSTM(input_dim = input_dim, output_dim = 300, return_sequences = True))  
	model.add(LSTM(input_dim = 300, output_dim = 500, return_sequences = True))  
	model.add(Dropout(0.2))
	model.add(LSTM(input_dim = 500, output_dim = 200, return_sequences = False))  
	model.add(Dropout(0.2))
	model.add(Dense(input_dim = 200, output_dim = output_dim)) 
	model.add(Activation('sigmoid'))

	model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

	return model

def get_team_names(team_name_data):

	team_names = []

	for i in range(0, len(team_name_data)):
		if team_name_data[i] not in team_names:
			team_names.append(team_name_data[i])

	return team_names

def get_target_date():
	start_date = input("Please type the start date for which you would like to get (for example: 20160201):")
	start_date = int(start_date)
	end_date = input("Please type the end date for which you would like to get (for example: 20160228):")
	end_date = int(end_date)

	return start_date, end_date

def normalize_dataset(dataset):
	dataset = dataset.astype('float32')

	# normalize the dataset
	scaler = MinMaxScaler(feature_range=(0, 1))	
	dataset = numpy.delete(dataset, [1], axis=1)
	dataset = scaler.fit_transform(dataset)

	return dataset

def load_dataset():
	return pandas.read_csv(
		'seeds/inputs.csv',
		usecols = [1, 3, 4, 7, 8, 9, 10, 11, 14, 15, 16, 19, 20],
		engine = 'python'
	).values

def save_model(model, date):
	model.save(
		'saveModels/ModelNormal-' + str(date) + '.h5'
	)
	print('The trained Model saved successfully!')

	return

def result_report(result_data, past_win_loss, date):

	Consensus_Percent, WinLoss_Percent, winloss_array = get_percentage(result_data)

	winloss = [[x] for x in winloss_array]
	modified_testPredict = numpy.concatenate((result_data, winloss), axis=1)

	result1 = [[x] for x in past_win_loss]
	modified_testPredict = numpy.concatenate((numpy.array(modified_testPredict), result1), axis = 1)

	print ('--------------The prediction Result for date', date, '---------------')
	print (modified_testPredict)
	print ('The percentage of Consensus Prediction for date', date, ' was ', Consensus_Percent, '%.')
	print ('The percentage of WinLoss Prediction for date', date, ' was ', WinLoss_Percent, '%.')

	# write the prediction result to csv file named result.csv
	filename = 'results/resultNormalWithVPT.csv'
	with open(filename, "wb") as f:
		numpy.savetxt(f, testPredict, delimiter=",", fmt="%s")

	return

def train_model(model, train_x, train_y):
	early_stopping = EarlyStopping(
		monitor = 'val_loss',
		min_delta = 0.0000001,
		patience = 2,
		verbose = 0,
		mode = 'max'
	)

	return model.fit(
		train_x,
		train_y,
		nb_epoch=200,
		batch_size=1,
		shuffle=True,
		show_accuracy=True,
		verbose=1,
		validation_split=0.1,
		callbacks=[early_stopping]
	)

def model_predict(model, test_x, point_diff_max, point_diff_min):

	testPredict = model.predict(test_x, batch_size=1, verbose=1)

	return testPredict * (point_diff_max - point_diff_min) + point_diff_min

# fix random seed for reproducibility
numpy.random.seed(20)

dataset = load_dataset()

start_date, end_date = get_target_date()

teams = get_team_names(dataset[:, 3])
original_dataset = dataset
dataset, teamnames = features_extract(dataset, teams)
dataset_clone = dataset
dataset = normalize_dataset(dataset)
model = build_model(
	input_dim = 67,
	output_dim = 1
)

for select_date in range(start_date, end_date):	

	trainX, trainY, testX, testY, testTeams = build_train_test_datasets(dataset_clone, dataset, select_date, teamnames)

	train_model(model, trainX, trainY)
	#loss, accuracy = model.evaluate(trainX, trainY)

	# point_diff_max = numpy.amax(dataset_clone[:, 4])
	# point_diff_min = numpy.amin(dataset_clone[:, 4])

	point_diff_max = 10
	point_diff_min = -10

	testPredict = model_predict(model, testX, point_diff_max, point_diff_min)
	past_win_loss = get_confidence(original_dataset, testPredict, select_date)

	result = [[int(numpy.sign(x))] for x in list(testPredict)]
	testPredict = numpy.concatenate(
		(
			numpy.array(testTeams),
		 	result
		),
		axis = 1
	)
	result_report(testPredict, past_win_loss, select_date)
	save_model(model, select_date)
