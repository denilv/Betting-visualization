# LSTM for predicting winners with pick and vegas - normal algorithm
import numpy
import pandas
import math
import time

# from keras.utils.visualize_util import model_to_dot
from datetime import datetime


# convert an array of values into a dataset matrix
def normalize_team(dataset, rate):

	win_num = 0
	loss_num = 0
	errors_point_difference = 0
	for i in range(0, len(dataset)):
		point_differential = dataset[i, 0] - dataset[i, 1]
		if point_differential != dataset[i, 2]:
			 errors_point_difference += 1
		if dataset[i, 4] >= rate:
			if dataset[i, 3] == 1:
				win_num += 1
			elif dataset[i, 3] == 0:
				loss_num += 1
	print 'errors_point_difference', errors_point_difference
	return win_num, loss_num

# load the dataset
dataframe = pandas.read_csv('seeds/backtesting2016.csv', usecols = [5, 6, 7, 8, 9], engine = 'python')
dataset = dataframe.values

rate = input ("Please type the percent (for example: 0.7):")
rate = float (rate)

win, loss = normalize_team(dataset, rate)

print 'The win is ', win, ' loss is ', loss