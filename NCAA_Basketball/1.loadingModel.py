# LSTM for predicting winners with pick and vegas - normal algorithm
import numpy
import matplotlib.pyplot as plt
import pandas
import math
import time
import copy
from IPython.display import SVG
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Activation, Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.callbacks import EarlyStopping
from keras.layers.embeddings import Embedding
# from keras.utils.visualize_util import plot
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from IPython.display import SVG
# from keras.utils.visualize_util import model_to_dot
from datetime import datetime

teams = []
def sigmoid(c, x):
  return 1 / (1 + math.exp(-c * x))

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
		# opponent_pick = 0
		# if i < len(dataset) - 1:
		# 	opponent_pick = dataset[i + 1, 10]
		# else:
		# 	opponent_pick = 0
		team_matrix.append(numpy.concatenate((team, opponent_team, [time.split(':', 1)[0]])))
		team_names.append([teamname, opponteamname, consensus_sp, spread, point, opponent_point])
	dataset = numpy.concatenate((dataset, team_matrix), axis=1)
	dataset = numpy.delete(dataset, [1, 3, 4, 5], axis = 1)
	dataset = dataset[::2]
	team_names = team_names[::2]

	return numpy.array(dataset), numpy.array(team_names)

def getConfidence(testTeams, d_set, testPredict, select_date):
	win_loss = []
	d_set = d_set[::2]
	home_margin = 0
	num_date = numpy.searchsorted(d_set[:,2], select_date)
	trainLength = num_date
	num_date = [i for i,val in enumerate(d_set[:,2]) if val == select_date]
	testLength = num_date[-1]
	k = 0
	for i in range(trainLength, testLength + 1):
		team1_pos, team1_nav, team2_pos, team2_nav = 0, 0, 0, 0
		for j in range(trainLength - 2000, trainLength):
			if d_set[j][4] == d_set[i][4]:
				diff = numpy.subtract(numpy.int_(d_set[j, 6]), numpy.int_(d_set[j, 7]))
				if diff > 0:
					team1_pos += 1
				else:
					team1_nav += 1
			if d_set[j][4] == d_set[i][5]:
				diff = numpy.subtract(numpy.int_(d_set[j, 6]), numpy.int_(d_set[j, 7]))
				if diff > 0:
					team2_pos += 1
				else:
					team2_nav += 1
		alpha = (team1_pos - team1_nav) - (team2_pos - team2_nav + home_margin)
		# print '-----alpha------', alpha
		# print '-----d_set[i][8]------', d_set[i][8]
		# print '-----testPredict[k][0]------', testPredict[k][0]
		# print '-----x------', d_set[i][8] + testPredict[k][0] - alpha * 0.8
		x = sigmoid(0.1, d_set[i][8] + testPredict[k][0] - alpha * 0.8)
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
def mapping(name):
	# training dataset generation
	# teams = [ 'DAL', 'WAS', 'POR', 'NYK', 'SAC', 'DET', 'ATL', 'NOP', 'LAC', 'DEN', 'PHI', 'LAL', 'CLE', 'CHI', 'IND', 'TOR', 'ORL', 'MIA', 'MEM', 'BOS', 'MIL', 'BKN', 'OKC', 'HOU', 'PHX', 'MIN', 'UTA', 'GSW', 'SAS', 'CHA', 'EAST', 'WEST', 'WALLSTAR', 'USALLSTAR' ]
	normalized_values = numpy.eye(len(teams))
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

	startTrain = numpy.searchsorted(dataset[:,1], date - 1)
	if date == 20170301:
		startTrain = numpy.searchsorted(dataset[:,1], 20170228)
	num_date = numpy.searchsorted(dataset[:,1], date)
	trainLength = num_date
	num_date = [i for i,val in enumerate(dataset[:,1]) if val == date]
	testLength = num_date[-1] + 1
	return startTrain, trainLength, testLength

# fix random seed for reproducibility
numpy.random.seed(20)

# load the dataset
dataframe = pandas.read_csv('seeds/inputs.csv', usecols = [1, 2, 3, 4, 7, 8, 9, 10, 11, 12, 14, 19, 20, 21], engine = 'python')
dataset = dataframe.values

select_date = input ("Please type the date for which you would like to get (for example: 20160210):")
select_date = int (select_date)

for i in range(0, len(dataset[:, 4])):
	if dataset[:, 4][i] not in teams:
		teams.append(dataset[:, 4][i])
len_teamName = len(teams)

original_dataset = dataset
dataset, teamnames = normalize_team(dataset)
dataset_clone = dataset

# get maximum of point difference
point_diff_max = numpy.amax(dataset[:, 4])
point_diff_min = numpy.amin(dataset[:, 4])

# get maximum of spread
spread_max = numpy.amax(dataset[:, 5])
spread_min = numpy.amin(dataset[:, 5])

dataset = dataset.astype('float32')

# normalize the dataset
scaler = MinMaxScaler(feature_range = (0, 1))
dataset = scaler.fit_transform(dataset)

dataset = numpy.delete(dataset, [1], axis = 1)

# create and fit the LSTM network
dim_inputs = 700
dim_outputs = 1

mem_blocks = 2
tt_batch_size = 1
tp_batch_size = 1

# create and fit the LSTM network
model = Sequential()
model_date = str(select_date - 1)
if select_date == 20170301:
	model_date = str(20170228)
model = load_model('saveModels/ModelNormal-' + model_date + '.h5')

# split into train and test sets
startTrain, trainLength, testLength = getLength(dataset_clone, select_date)

train, test = dataset[startTrain:trainLength,:], dataset[trainLength:testLength,:]
trainX, trainY = create_dataset(train)
testX, testY = create_dataset(test)

testTeams = teamnames[trainLength:testLength,:]

# reshape input to be [samples, time steps, features]
trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

tt_nb_epoch = 200

# training NN Model
early_stopping = EarlyStopping(monitor = 'val_loss', min_delta = 0.0000001, patience = 2, verbose = 0, mode = 'max')
hist = model.fit(
	trainX,
	trainY,
	nb_epoch = tt_nb_epoch,
	batch_size = tt_batch_size,
	shuffle = True,
	show_accuracy = True,
	verbose = 1,
	validation_split = 0.1,
	callbacks = [early_stopping]
)

# evaluate the network
loss, accuracy = model.evaluate(trainX, trainY)
# print "\nLoss: %.2f, Accuracy: %.2f%%" % (loss, accuracy * 100)
# print ("\nLoss: %.2f, Accuracy: %.2f%%" % (loss, accuracy * 100))

# make predictions
testPredict = model.predict(testX, batch_size = tp_batch_size, verbose = 1)

testPredict = testPredict * (point_diff_max - point_diff_min) + point_diff_min

result = [[int(numpy.around(x, decimals = 0))] for x in testPredict]

past_win_loss = getConfidence(testTeams, original_dataset, testPredict, select_date)

testPredict = numpy.concatenate((numpy.array(testTeams), result), axis = 1)

NN_Percent, Consensus_Percent, WinLoss_Percent, winloss_array, Winner_array = getPercentage(testPredict)

winloss = [[x] for x in winloss_array]
modified_testPredict = numpy.concatenate((testPredict, winloss), axis = 1)

winloss1 = [[x] for x in Winner_array]
modified_testPredict = numpy.concatenate((modified_testPredict, winloss1), axis = 1)

winloss = [[x] for x in winloss_array]
modified_testPredict = numpy.concatenate((testPredict, winloss), axis = 1)

result1 = [[x] for x in past_win_loss]
modified_testPredict = numpy.concatenate((numpy.array(modified_testPredict), result1), axis = 1)

print '--------------The prediction Result for date', select_date, '---------------'
print modified_testPredict
print 'The percentage of Result for date', select_date, ' was ', NN_Percent, '%.'
print 'The percentage of Consensus Prediction for date', select_date, ' was ', Consensus_Percent, '%.'

# write the prediction result to csv file named result.csv
filename = 'results/resultStepByStepWithVPT-' + str(select_date) + '.csv'
with open(filename, "wb") as f:
	numpy.savetxt(f, testPredict, delimiter=",", fmt="%s")

# save Model
filename = 'saveModels/ModelNormal-' + str(select_date) + '.h5'
model.save(filename)
print 'The trained Model saved successfully!'