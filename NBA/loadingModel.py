# LSTM for predicting winners with pick and vegas - weekly algorithm
import numpy
import matplotlib.pyplot as plt
import pandas
import math
import time
from keras.models import Sequential
from keras.layers import Dense
from keras.models import load_model
from keras.layers import LSTM
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

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
		diff = numpy.subtract(numpy.int_(point), numpy.int_(opponent_point));
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
		team_names.append([teamname, opponteamname, consensus_sp, point, opponent_point])
	dataset = numpy.concatenate((dataset, team_matrix), axis=1)
	dataset = numpy.delete(dataset, [3, 4, 5], axis = 1)
	dataset = dataset[::2]
	team_names = team_names[::2]
	return numpy.array(dataset), numpy.array(team_names)

# convert an array of values into a dataset matrix
def create_dataset(data):
	dataX, dataY, team_matrix = [], [], []
	dataX = numpy.delete(data, [3, 4], axis = 1)
	for i in range(0, len(data)):
		dataY.append([data[i, 3], data[i, 4]])
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
	data = numpy.array(data)
	for i in range(0, len(data)):
		diff1 = numpy.sign(numpy.subtract(numpy.int_(data[i, 3]), numpy.int_(data[i, 4])));
		diff2 = numpy.sign(numpy.subtract(numpy.int_(data[i, 5]), numpy.int_(data[i, 6])));	
		if diff1 == diff2:
			percent1 = percent1 + 1
		if data[i, 2] == 1:
			percent2 = percent2 + 1
	value1 = (percent1 / float(len(data))) * 100
	value2 = (percent2 / float(len(data))) * 100
	return numpy.around(value1, decimals = 2), numpy.around(value2, decimals = 2)

# fix random seed for reproducibility
numpy.random.seed(7)

# load the dataset
dataframe = pandas.read_csv('seeds/inputs.csv', usecols = [1, 2, 3, 4, 7, 8, 9, 10, 12, 14, 19, 20, 21], engine = 'python')
dataset = dataframe.values

dataset, teamnames = normalize_team(dataset)

trainX = []
trainName = []
testX = []
temp = []
tempname = []
for i in range(0, 13):
	for j in range(0, len(dataset)):
		if dataset[j, 0] != 2016:
			if i == 0:
				temp.append(dataset[j])
				tempname.append(teamnames[j])
		else:
			if dataset[j, 1] == i:
				temp.append(dataset[j])
				tempname.append(teamnames[j])
	trainX.append(temp)
	trainName.append(tempname)
	temp = []
	tempname = []

# create and fit the LSTM network
model = Sequential()
model = load_model('saveModels/ModelWeeklyWithVPT.h5')

# split into train and test sets
testDataX = []
for i in range(0, 12):
	testDataX.append(trainX[i + 1])

# select month to be predicted
i = 11

# get maximum points and opponent points
temp_train = numpy.array(trainX[i])
point_max = numpy.amax(temp_train[:, 3])
opponent_point_max = numpy.amax(temp_train[:, 4])

scaler = MinMaxScaler(feature_range = (0, 1))
testDataX[i] = scaler.fit_transform(testDataX[i])

testX, testY = create_dataset(testDataX[i])

# reshape input to be [samples, time steps, features]
testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

# make predictions
testPredict = model.predict(testX)

result = [[int(round(x[0] * point_max)), int(round(x[1] * opponent_point_max))] for x in testPredict]
testPredict = numpy.concatenate((numpy.array(trainName[i + 1]), result), axis = 1)

print '--------------The prediction Result for week', i + 1 ,'---------------'
print testPredict
print 'The percentage of Result for week', i + 1, ' was ', getPercentage(testPredict), '%.'

# write the prediction result to csv file named result.csv
filename = 'results/ResultWeeklyWithVPT-' + str(i + 1) + '-' + time.strftime("%Y%m%d-%H%M%S") + '.csv'
with open(filename, "wb") as f:
	numpy.savetxt(f, testPredict, delimiter=",", fmt="%s")