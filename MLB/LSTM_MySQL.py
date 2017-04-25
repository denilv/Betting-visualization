# LSTM for predicting winners with pick and vegas - weekly algorithm

# Server: sql6.freemysqlhosting.net
# Name: sql6151060
# Username: sql6151060
# Password: 6SqGWfrHKu
# Port number: 3306

import numpy
import matplotlib.pyplot as plt
import pandas
import math
import time
import copy
import pymysql
from IPython.display import SVG
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.callbacks import EarlyStopping
from keras.layers.embeddings import Embedding
from keras.utils.visualize_util import plot
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from IPython.display import SVG
from keras.utils.visualize_util import model_to_dot

# convert an array of values into a dataset matrix
def normalize_team(dataset):
	team_matrix = []
	team_names = []

	for i in range(0, len(dataset)):
		season = int(dataset[i, 0])
		week = int(dataset[i, 1])
		date = int(dataset[i, 2])
		time = dataset[i, 3]
		team = mapping(dataset[i, 4])
		opponent_team = mapping(dataset[i, 5])
		teamname = dataset[i, 4]
		opponteamname = dataset[i, 5]
		point = int(dataset[i, 6])
		opponent_point = int(dataset[i, 7])
		spread = float(dataset[i, 8])
		point_diff = numpy.subtract(numpy.int_(point), numpy.int_(opponent_point))
		diff = numpy.add(point_diff, spread)
		consensus_sp = 1
		if int(dataset[i, 11]) >= 50:
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
			opponent_pick = int(dataset[i + 1, 10])
		else:
			opponent_pick = 0
		team_matrix.append(numpy.concatenate((team, opponent_team, [opponent_pick])))
		team_names.append([season, week, date, teamname, opponteamname, consensus_sp, spread, point, opponent_point])
	dataset = numpy.concatenate((dataset, team_matrix), axis=1)
	dataset = numpy.delete(dataset, [2, 3, 4, 5], axis = 1)
	dataset = dataset[::2]
	team_names = team_names[::2]
	return numpy.array(dataset), numpy.array(team_names)

# convert an array of values into a dataset matrix
def create_dataset(data):
	dataX, dataY, team_matrix = [], [], []
	dataX = numpy.delete(data, [2, 3], axis = 1)
	for i in range(0, len(data)):
		dataY.append([data[i, 2], data[i, 3]])
	return numpy.array(dataX), numpy.array(dataY)

# standardize name of teams
def mapping(name):
	# training dataset generation
	teams = ['DEN', 'CAR', 'ARI', 'NE', 'PIT', 'SEA', 'GB', 'KC', 'WAS', 'MIN', 'CIN', 'HOU', 'TB', 'SF', 'STL', 'SD', 'OAK', 'CHI', 'DET', 'NYG', 'PHI', 'DAL', 'TEN', 'IND', 'JAC', 'CLE', 'BAL', 'ATL', 'NO', 'MIA', 'NYJ', 'LA', 'BUF']
	normalized_values = numpy.eye(33);
	index = teams.index(name);
	return normalized_values[index]

# get Percentage of weekly prediction
def getPercentage(data):
	percent1 = 0
	percent2 = 0
	result_winloss_array = []
	data = numpy.array(data)
	for i in range(0, len(data)):
		point_diff1 = numpy.subtract(numpy.int_(data[i, 7]), numpy.int_(data[i, 8]))
		diff1 = numpy.sign(numpy.add(point_diff1, numpy.float_(data[i, 6])))
		point_diff2 = numpy.subtract(numpy.int_(data[i, 9]), numpy.int_(data[i, 10]))
		diff2 = numpy.sign(numpy.add(point_diff2, numpy.float_(data[i, 6])))

		if diff1 == diff2:
			percent1 = percent1 + 1
			result_winloss_array.append(1)
		else:
			result_winloss_array.append(0)
		if numpy.int_(data[i, 5]) == 1:
			percent2 = percent2 + 1
	value1 = (percent1 / float(len(data))) * 100
	value2 = (percent2 / float(len(data))) * 100
	return numpy.around(value1, decimals = 2), numpy.around(value2, decimals = 2), result_winloss_array

# insert the prediction result to database
def insertDatabase(data):
	for j in range(0, len(data)):
		array = numpy.array([""])
		array = numpy.append(array, data[j], axis = 0)

		try:
			sql = "INSERT INTO outputs VALUES %r;" % (tuple(array.tolist()),)
			cur.execute(sql)
			# Commit your changes in the database
			conn.commit()
		except:
			# Rollback in case there is any error
			conn.rollback()

# fix random seed for reproducibility
numpy.random.seed(1337)

# Connect to the database
conn = pymysql.connect(host = 'sql6.freemysqlhosting.net', port = 3306, user = 'sql6151060', passwd = '6SqGWfrHKu', db = 'sql6151060')
cur = conn.cursor()
cur.execute("SELECT * FROM inputs")
dataset = []
for row in cur.fetchall():
	dataset.append(numpy.array(row))
dataset = numpy.array(dataset)

dataset = dataset[:, [1, 2, 3, 4, 7, 8, 9, 10, 12, 14, 19, 20, 21]]

dataset, teamnames = normalize_team(dataset)
dataset = dataset.astype(numpy.float)

# get maximum points and opponent points
point_max = numpy.amax(dataset[:, 2])
opponent_point_max = numpy.amax(dataset[:, 3])

# normalization for inputs
scaler = MinMaxScaler(feature_range = (0, 1))
dataset[:, [ 2, 3, 4, 5, 6, 7, 8, 75]] = scaler.fit_transform(dataset[:, [ 2, 3, 4, 5, 6, 7, 8, 75]])

dataset_temp = copy.copy(dataset)
dataset[:, [0, 1]] = scaler.fit_transform(dataset[:, [0, 1]])

trainX = []
trainName = []
testX = []
temp = []
tempname = []
for i in range(0, 15):
	for j in range(0, len(dataset)):
		if dataset_temp[j, 0] != 2016:
			if i == 0:
				temp.append(dataset[j])
				tempname.append(teamnames[j])
		else:
			if dataset_temp[j, 1] == i:
				temp.append(dataset[j])
				tempname.append(teamnames[j])
	trainX.append(temp)
	trainName.append(tempname)
	temp = []
	tempname = []

# create and fit the LSTM network
dim_inputs = 74
max_features = 74
embedding_vecor_length = 74
max_review_length = 100
dim_outputs = 2

# For Python 2.7 version
tt_nb_epoch = 15
tt_batch_size = 1
tp_batch_size = 1
mem_blocks = 2000

# # For Python 3.5 version
# tt_nb_epoch = 2
# tt_batch_size = 32
# tp_batch_size = 32
# mem_blocks = 2

model = Sequential()
# model.add(LSTM(mem_blocks, input_dim = dim_inputs))
# # model.add(Embedding(dim_inputs, dim_outputs, init = 'uniform', input_length = None));
# model.add(Dropout(0.2))
# model.add(Dense(dim_outputs, activation = 'sigmoid'))
# model.compile(loss = 'mean_squared_error', optimizer = 'adam', metrics = ['accuracy'])

model.add(LSTM(input_dim = dim_inputs, output_dim = 300, return_sequences = True))
model.add(LSTM(input_dim = 300, output_dim = 500, return_sequences = True))
model.add(Dropout(0.2))
model.add(Dense(input_dim = 200, output_dim = dim_outputs))
model.add(Activation('sigmoid'))
model.compile(loss = 'mean_squared_error', optimizer = 'adam', metrics = ['accuracy'])  

# split into train and test sets
testDataX = []
for i in range(0, 14):
	testDataX.append(trainX[i + 1])

NN_Percents = []
Consensus_Percents = []

for i in range(0, 14):

	# get maximum points and opponent points
	# temp_train = numpy.array(trainX[i])
	# point_max = numpy.amax(temp_train[:, 3])
	# opponent_point_max = numpy.amax(temp_train[:, 4])

	# # normalize the dataset
	# scaler = MinMaxScaler(feature_range = (0, 1))
	# trainX[i] = scaler.fit_transform(trainX[i])
	# scaler = MinMaxScaler(feature_range = (0, 1))
	# testDataX[i] = scaler.fit_transform(testDataX[i])

	trainXX, trainYY = create_dataset(numpy.array(trainX[i]))
	testX, testY = create_dataset(numpy.array(testDataX[i]))

	# reshape input to be [samples, time steps, features]
	trainXX = numpy.reshape(trainXX, (trainXX.shape[0], 1, trainXX.shape[1]))
	testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

	tt_nb_epoch = 20
	if i == 0:
		tt_nb_epoch = 100

	# training NN Model

	early_stopping = EarlyStopping(monitor = 'val_loss', min_delta = 0.000001, patience = 2, verbose = 0, mode = 'max')
	hist = model.fit(
		trainXX,
		trainYY,
		nb_epoch = tt_nb_epoch,
		batch_size = tt_batch_size,
		shuffle = True,
		show_accuracy = True,
		verbose = 1,
		validation_split = 0.1,
		callbacks = [early_stopping]
	)
	print hist.history
	# print (hist.history)

	# make predictions
	testPredict = model.predict(testX, batch_size = tp_batch_size, verbose = 1)

	result = [[int(round(x[0] * point_max)), int(round(x[1] * opponent_point_max))] for x in testPredict]
	testPredict = numpy.concatenate((numpy.array(trainName[i + 1]), result), axis = 1)

	NN_Percent, Consensus_Percent, winloss_array = getPercentage(testPredict)

	winloss = [[x] for x in winloss_array]
	modified_testPredict = numpy.concatenate((testPredict, winloss), axis = 1)

	NN_Percents.append(NN_Percent)
	Consensus_Percents.append(Consensus_Percent)

	# # For Python 3.5 version
	# print ('--------------The prediction Result for week', i + 1 ,'---------------')
	# print (modified_testPredict)
	# print ('The percentage of Result for week', i + 1, ' was ', NN_Percent, '%.')
	# print ('The percentage of Consensus Prediction for week', i + 1, ' was ', Consensus_Percent, '%.')

	# For Python 2.7 version
	print '--------------The prediction Result for week', i + 1 ,'---------------'
	print modified_testPredict
	print 'The percentage of Result for week', i + 1, ' was ', NN_Percent, '%.'
	print 'The percentage of Consensus Prediction for week', i + 1, ' was ', Consensus_Percent, '%.'
	
	if i == 13:
		insertDatabase(modified_testPredict)

	# write the prediction result to csv file named result.csv
	filename = 'results/ResultWeeklyWithVPT.csv'
	with open(filename, "wb") as f:
		numpy.savetxt(f, modified_testPredict, delimiter=",", fmt="%s")

# save Model
filename = 'saveModels/ModelWeeklyWithVPT.h5'
model.save(filename)

# # For Python 3.5 version
# print ('The trained Model saved successfully!')
# print ('The NN Prediction percents are ', NN_Percents)
# print ('The Consensus Prediction percents are ', Consensus_Percents)

# For Python 2.7 version
print 'The trained Model saved successfully!'
print 'The NN Prediction percents are ', NN_Percents
print 'The Consensus Prediction percents are ', Consensus_Percents

plot(model, to_file='results/model.png', show_shapes = True, show_layer_names = True)
SVG(model_to_dot(model).create(prog='dot', format='svg'))

# close connection of MySQL
cur.close()
conn.close()