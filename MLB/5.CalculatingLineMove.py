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

global team, results
teams = []
results = []

def signWithMargin(x):
	if x > 0.1:
		return 1
	elif x < -0.1:
		return -1
	else:
		return 0

def normalizeLineMovements(array):
	newList = []
	print 'array', array
	for x in array:
		
		# print 'x[2]', x[2]
		if float(x[2]) > 10.0:
			print ('here1')
			newList.append([x[0], x[1], 1.0])
		elif float(x[2]) < -10:
			print ('here2')
			newList.append([x[0], x[1], -1.0])
		else:
			print ('here3')
			newList.append([x[0], x[1], float(x[2]) / 10.0])
	print 'newList', newList
	return newList

# get the good chance from line movement
def getChancefromLineMovement(item, data_spread):

	datetimeFormat = '%m%d%H:%M%p'
	item[3] = item[3][:-1] + 'PM'
	str_startDate = str(item[2])[4:] + item[3]
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
		return -104.05

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
				datetimeFormat1 = '%m/%d %H:%M %p'
				
				if str(data[i, 5])[:-9] == '02/29' or str(data[0, 5])[:-9] == '02/29':
					continue;
				tdelta = datetime.strptime (data[0, 5], datetimeFormat1) - datetime.strptime (data[i, 5], datetimeFormat1)
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

		# print ('time_array_pin', time_array_pin)

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

			# print ('del_pin', spread_array_pin[i] - spread_array_pin[i - 1])
			# print ('del_book', spread_array_pin_inter_book[i] - spread_array_pin_inter_book[i - 1])
			# print ('del_sports', spread_array_pin_inter_sports[i] - spread_array_pin_inter_sports[i - 1])
			# print ('del_bovada', spread_array_pin_inter_bovada[i] - spread_array_pin_inter_bovada[i - 1])
			# print ('------------------')

			# print ('sign_del_pin', sign_del_pin)
			# print ('sign_del_book', sign_del_book)
			# print ('sign_del_sports', sign_del_sports)
			# print ('sign_del_bovada', sign_del_bovada)
			# print ('------------------')

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
		# print ("------result------", result)
		return result

def getTimeRange (day1, day2, data):
	 return numpy.searchsorted(data[:,2], day1)-1, numpy.searchsorted(data[:,2], day2+1)-1
# load the dataset
dataframe = pandas.read_csv('seeds/inputs.csv', usecols = [1, 2, 3, 4, 7, 8, 9, 10, 11, 12, 14, 19, 20, 21], engine = 'python')
dataset = dataframe.values

start_date = input ("Please type the start date for which you would like to get (for example: 20170210):")
start_date = int (start_date)
end_date = input ("Please type the end date for which you would like to get (for example: 20170213):")
end_date = int (end_date)

dataframe_spread = pandas.read_csv ('seeds/ncaa_movement.csv', usecols = [0, 1, 2, 5, 6, 7, 8, 9], engine = 'python')
dataset_spread = dataframe_spread.values

print ('Calculating Line Movement history...')

start, end = getTimeRange(start_date, end_date, dataset)

print 'start_number', start
print 'end_number', end

lineMovementParams = [[0, 0, 0]] * len(dataset)
old_predict = 0
print 'lineMovementParams', lineMovementParams
for i in range(start, end):
	# print '--i--', i
	predicted_spread = getChancefromLineMovement(dataset[i], dataset_spread)
	print 'predicted_spread', predicted_spread
	print 'dataset[i, 4]', dataset[i, 4]
	print 'dataset[i, 5]', dataset[i, 5]
	print 'old_predict', old_predict

	if predicted_spread != 0 and predicted_spread != -104.05:
		lineMovementParams[i] = [dataset[i, 4], dataset[i, 5], predicted_spread]
	elif predicted_spread == -104.05:
		lineMovementParams[i] = [dataset[i, 4], dataset[i, 5], old_predict]
	else:
		lineMovementParams[i] = [dataset[i, 4], dataset[i, 5], 0]
	old_predict = -predicted_spread

lineMovementParams = normalizeLineMovements(numpy.array(lineMovementParams))

# write the prediction result to csv file named result.csv
filename = 'results/lineMovement.csv'
with open(filename, "wb") as f:
	numpy.savetxt(f, lineMovementParams, delimiter=",", fmt="%s")