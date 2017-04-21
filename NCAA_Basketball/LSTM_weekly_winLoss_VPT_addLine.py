# LSTM for predicting winners with pick and vegas - weekly algorithm
import numpy
import matplotlib.pyplot as plt
import pandas
import math
import time
import copy
import statsmodels.api as sm
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
		point_diff = dataset[i, 8]
		diff = numpy.add(point_diff, spread)
		consensus_sp = 1
		if dataset[i, 12] >= 50:
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
			opponent_pick = dataset[i + 1, 11]
		else:
			opponent_pick = 0
		team_matrix.append(numpy.concatenate((team, opponent_team, [time.split(':', 1)[0], opponent_pick])))
		team_names.append([teamname, opponteamname, consensus_sp, spread, point, opponent_point])
	dataset = numpy.concatenate((dataset, team_matrix), axis=1)
	dataset = numpy.delete(dataset, [2, 3, 4, 5], axis = 1)
	dataset = dataset[::2]
	team_names = team_names[::2]
	return numpy.array(dataset), numpy.array(team_names)

# convert an array of values into a dataset matrix
def create_dataset(data):
	dataX, dataY, team_matrix = [], [], []
	dataX = numpy.delete(data, [2, 3, 4], axis = 1)
	for i in range(0, len(data)):
		dataY.append([data[i, 4]])
	return numpy.array(dataX), numpy.array(dataY)

# standardize name of teams
def mapping(name):
	# training dataset generation
	teams = ['DEN', 'CAR', 'ARI', 'NE', 'PIT', 'SEA', 'GB', 'KC', 'WAS', 'MIN', 'CIN', 'HOU', 'TB', 'SF', 'SD', 'OAK', 'CHI', 'DET', 'NYG', 'PHI', 'DAL', 'TEN', 'IND', 'JAC', 'CLE', 'BAL', 'ATL', 'NO', 'MIA', 'NYJ', 'LA', 'BUF']
	normalized_values = numpy.eye(32);
	index = teams.index(name);
	return normalized_values[index]

# get Percentage of weekly prediction
def getPercentage(data, confidence, non_confidence):
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
			confidence[data[i, 0]] = confidence[data[i, 0]] + 1;
			confidence[data[i, 1]] = confidence[data[i, 1]] + 1;
		else:
			result_winloss_array.append(0)
			non_confidence[data[i, 0]] = non_confidence[data[i, 0]] + 1;
			non_confidence[data[i, 1]] = non_confidence[data[i, 1]] + 1;

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
	return numpy.around(value1, decimals = 2), numpy.around(value2, decimals = 2), numpy.around(value3, decimals = 2), result_winloss_array, result_winloss_array2, confidence, non_confidence

# plot the prediction accuracy of both Neural Network and Consensus
def plotAccuracy(NN_Percents, Consensus_Percents):
	year = ['Week 1', 'Week 2', 'Week 3', 'Week 4', 'Week 5', 'Week 6', 'Week 7', 'Week 8', 'Week 9', 'Week 10', 'Week 11', 'Week 12', 'Week 13', 'Week 14', 'Week 15']
	N = len(NN_Percents)
	ind = numpy.arange(N)
	width = 0.35

	fig, ax = plt.subplots()
	fig.set_size_inches(18.5 * 0.9, 10.5 * 0.9)
	rects1 = ax.bar(ind, NN_Percents, width, color = 'r')
	rects2 = ax.bar(ind+width, Consensus_Percents, width, color='b')

	ax.set_ylabel('Accuracy, %')
	ax.set_title('Prediction Accuracy For 2016 week 1 ~ 15')
	ax.set_xticks(ind + (width * 0.5))
	ax.set_xticklabels(year)

	ax.legend((rects1[0], rects2[0]), ("Neural Network Prediction", "Consensus"))
	ax.set_ylim([min(min(NN_Percents), min(Consensus_Percents)) * 0.97, max(max(NN_Percents), max(Consensus_Percents)) * 1.03])

	plt.savefig('results/accuracy_chart.png', bbox_inches = 'tight', dpi = 100)

def plotPrediction(loss_array, accuracy_array):

	fig, ax = plt.subplots()

	year = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
	y1 = plt.plot(year, loss_array, label = 'Training Accuracy', color = 'blue')
	y2 = plt.plot(year, accuracy_array, label = 'Loss Curve', color = 'red')

	fig.suptitle('Training Accuracy, Loss Accuracy')
	plt.xlabel('Weeks')
	plt.ylabel('Accuracy')

	ax.set_ylim([0, 1])
	handles, labels = ax.get_legend_handles_labels()
	ax.legend(handles, labels)
	plt.savefig('results/TrainingAccuracyChart.png', bbox_inches = 'tight', dpi = 100)

def predictPointSpread(item, data_spread):

	datetimeFormat = '%m%d%H:%M%p'
	item[3] = item[3][:-1] + 'PM'
	str_startDate = str(item[2])[4:] + item[3]
	startTime = datetime.strptime(str_startDate, datetimeFormat)
	
	data = []
	for i in range(0, len(data_spread)):
		if data_spread[i, 0] == 'pinnacle' and item[2] == data_spread[i, 1] and item[4] == data_spread[i, 3] and item[5] == data_spread[i, 4]:
			data.append(data_spread[i])

	if not data:
		return 0

	else:
		data = numpy.array(data)
		time_array = []
		spread_array = []

		for i in range(0, len(data)):
			datetimeFormat = '%m/%d %H:%M %p'
			tdelta = datetime.strptime(data[i, 5], datetimeFormat) - datetime.strptime(data[len(data) - 1, 5], datetimeFormat)
			tdelta = int(tdelta.total_seconds () / 60)
			spread = data[i, 6].split ( )[0]
			if spread != 'PK':
				time_array.append (tdelta)
				spread_array.append (float(spread))
				# print ('\nt =', tdelta, 'spread =', spread)
		t_final = startTime - datetime.strptime (data[len(data) - 1, 5], datetimeFormat);
		t_final = int(t_final.total_seconds() / 60)
		t_final = numpy.array (t_final)
		olsmod = sm.OLS(spread_array, time_array)
		olsres = olsmod.fit()

		ypred = olsres.predict(t_final)
		return ypred[0]

select_week = input("[VEGAS] Please type the week for which you would like to get (1 ~ 16):")

# fix random seed for reproducibility
numpy.random.seed(1337)

# load the dataset
dataframe = pandas.read_csv('seeds/inputs.csv', usecols = [1, 2, 3, 4, 7, 8, 9, 10, 11, 12, 14, 19, 20, 21], engine = 'python')
dataset = dataframe.values

dataframe_spread = pandas.read_csv('seeds/nfl_move_pointspread_2016.csv', usecols = [0, 1, 2, 5, 6, 7, 8, 9], engine = 'python')
dataset_spread = dataframe_spread.values

for i in range(len(dataset)):
	predicted_spread = predictPointSpread(dataset[i], dataset_spread)
	if predicted_spread != 0:
		dataset[i, 8] = predicted_spread

dataset, teamnames = normalize_team(dataset)

# get maximum of point difference
point_diff_max = numpy.amax(dataset[:, 4])
point_diff_min = numpy.amin(dataset[:, 4])

# get maximum of spread
spread_max = numpy.amax(dataset[:, 5])
spread_min = numpy.amin(dataset[:, 5])

# normalization for inputs
scaler = MinMaxScaler(feature_range = (0, 1))
dataset[:, [2, 3, 4, 5, 6, 7, 8, 9, 74, 75]] = scaler.fit_transform(dataset[:, [2, 3, 4, 5, 6, 7, 8, 9, 74, 75]])

dataset_temp = copy.copy(dataset)
dataset[:, [0, 1]] = scaler.fit_transform(dataset[:, [0, 1]])

trainX = []
trainName = []
testX = []
temp = []
tempname = []
for i in range(0, select_week + 1):
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
dim_inputs = 73
max_features = 73
embedding_vecor_length = 73
max_review_length = 100
dim_outputs = 1

# For Python 2.7 version
tt_nb_epoch = 15
tt_batch_size = 1
tp_batch_size = 1
mem_blocks = 2500

# # For Python 3.5 version
# tt_nb_epoch = 2
# tt_batch_size = 32
# tp_batch_size = 32
# mem_blocks = 2

model = Sequential()

# model.add(LSTM(mem_blocks, input_dim = dim_inputs))
# model.add(Dropout(0.2))
# model.add(Dense(dim_outputs, activation = 'sigmoid'))
# model.compile(loss = 'mean_squared_error', optimizer = 'adam', metrics = ['accuracy'])

model.add(LSTM(input_dim = dim_inputs, output_dim = 300, return_sequences = True))  
model.add(LSTM(input_dim = 300, output_dim = 500, return_sequences = True))  
model.add(Dropout(0.2))
model.add(LSTM(input_dim = 500, output_dim = 200, return_sequences = False))  
model.add(Dropout(0.2))  
model.add(Dense(input_dim = 200, output_dim = dim_outputs))  
model.add(Activation('sigmoid'))
model.compile(loss = 'mean_squared_error', optimizer = 'adam', metrics = ['accuracy']) 

# split into train and test sets
testDataX = []
for i in range(0, select_week):
	testDataX.append(trainX[i + 1])

NN_Percents = []
Consensus_Percents = []
WinLoss_Percents = []
loss_array = []
accuracy_array = []
confidence = dict.fromkeys(['DEN', 'CAR', 'ARI', 'NE', 'PIT', 'SEA', 'GB', 'KC', 'WAS', 'MIN', 'CIN', 'HOU', 'TB', 'SF', 'SD', 'OAK', 'CHI', 'DET', 'NYG', 'PHI', 'DAL', 'TEN', 'IND', 'JAC', 'CLE', 'BAL', 'ATL', 'NO', 'MIA', 'NYJ', 'LA', 'BUF'], 0)
non_confidence = dict.fromkeys(['DEN', 'CAR', 'ARI', 'NE', 'PIT', 'SEA', 'GB', 'KC', 'WAS', 'MIN', 'CIN', 'HOU', 'TB', 'SF', 'SD', 'OAK', 'CHI', 'DET', 'NYG', 'PHI', 'DAL', 'TEN', 'IND', 'JAC', 'CLE', 'BAL', 'ATL', 'NO', 'MIA', 'NYJ', 'LA', 'BUF'], 0)
for i in range(0, select_week):

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

	# evaluate the network
	loss, accuracy = model.evaluate(trainXX, trainYY)
	print "\nLoss: %.2f, Accuracy: %.2f%%" % (loss, accuracy * 100)
	loss_array.append(loss)
	accuracy_array.append(accuracy)

	# make predictions
	testPredict = model.predict(testX, batch_size = tp_batch_size, verbose = 1)

	testPredict = testPredict * (point_diff_max - point_diff_min) + point_diff_min

	result = [[int(numpy.around(x, decimals = 0))] for x in testPredict]
	testPredict = numpy.concatenate((numpy.array(trainName[i + 1]), result), axis = 1)

	NN_Percent, Consensus_Percent, WinLoss_Percent, winloss_array, Winner_array, confidence, non_confidence = getPercentage(testPredict, confidence, non_confidence)

	winloss = [[x] for x in winloss_array]
	modified_testPredict = numpy.concatenate((testPredict, winloss), axis = 1)

	winloss1 = [[x] for x in Winner_array]
	modified_testPredict = numpy.concatenate((modified_testPredict, winloss1), axis = 1)

	NN_Percents.append(NN_Percent)
	Consensus_Percents.append(Consensus_Percent)
	WinLoss_Percents.append(WinLoss_Percent)

	# # For Python 3.5 version
	# print ('\n--------------The prediction Result for week', i + 1 ,'---------------')
	# print (modified_testPredict)
	# print ('\nThe percentage of Spread Point Result for week', i + 1, ' was ', NN_Percent, '%.')
	# print ('\nThe percentage of Consensus Prediction for week', i + 1, ' was ', Consensus_Percent, '%.')
	# print ('\nThe percentage of WinLoss for week', i + 1, ' was ', WinLoss_Percent, '%.')

	# For Python 2.7 version
	print '\n--------------The prediction Result for week', i + 1 ,'---------------'
	print modified_testPredict
	print '\nThe percentage of Result for week', i + 1, ' was ', NN_Percent, '%.'
	print '\nThe percentage of Consensus Prediction for week', i + 1, ' was ', Consensus_Percent, '%.'
	print '\nThe percentage of WinLoss for week', i + 1, ' was ', WinLoss_Percent, '%.'

	# write the prediction result to csv file named result.csv
	filename = 'results/ResultWeeklyWithVPT.csv'
	with open(filename, "wb") as f:
		numpy.savetxt(f, modified_testPredict, delimiter=",", fmt="%s")

print confidence
print non_confidence

# save Model
filename = 'saveModels/ModelWeeklyWithVPT.h5'
model.save(filename)

# # For Python 3.5 version
# print ('\nThe trained Model saved successfully!')
# print ('\nThe NN Prediction percents are ', NN_Percents, ' - ', round(numpy.average(NN_Percents), 2), '%')
# print ('\nThe Consensus Prediction percents are ', Consensus_Percents, ' - ', round(numpy.average(Consensus_Percents), 2), '%')
# print ('\nThe WinLoss Prediction percents are ', WinLoss_Percents, ' - ', round(numpy.average(WinLoss_Percents), 2), '%')
# print ('\nThe Confidence rates are ', accuracy_array, ' - ', round(numpy.average(accuracy_array) * 100, 2), '%')
# print ('\nThe Loss are ', loss_array, ' - ', round(numpy.average(loss_array) * 100, 2), '%')

# For Python 2.7 version
print '\nThe trained Model saved successfully!'
print '\nThe NN Prediction percents are ', NN_Percents, ' - ', round(numpy.average(NN_Percents), 2), '%'
print '\nThe Consensus Prediction percents are ', Consensus_Percents, ' - ', round(numpy.average(Consensus_Percents), 2), '%'
print '\nThe WinLoss Prediction percents are ', WinLoss_Percents, ' - ', round(numpy.average(WinLoss_Percents), 2), '%'
print '\nThe Confidence rates are ', accuracy_array, ' - ', round(numpy.average(accuracy_array) * 100, 2), '%'
print '\nThe Loss are ', loss_array, ' - ', round(numpy.average(loss_array) * 100, 2), '%'

# draw the structure of Neural Network
plot(model, to_file = 'results/model.png', show_shapes = True, show_layer_names = True)
SVG(model_to_dot(model).create(prog = 'dot', format = 'svg'))

# draw some figures to show the accuracies
plotAccuracy(NN_Percents, Consensus_Percents)
plotPrediction(loss_array, accuracy_array)