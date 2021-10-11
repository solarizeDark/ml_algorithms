import math
import os

import pandas as pd
from sklearn.linear_model import LinearRegression
import sklearn.metrics as skm
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as statistics

# y = b0 + b1 * x
# b1 = sum((xi-mean(x)) * (yi-mean(y))) / sum((xi – mean(x))^2)
# b0 = mean(y) – b1 * mean(x)

def r2_score(predicted, actual):
	n = 0
	d = 0
	ma = actual.mean()
	for p, a in zip(predicted, actual):
		n += (p - a) * (p - a)
		d += (a - ma) * (a - ma)
	return 1 - n / d

def rmse(predicted, actual):
	sum = 0
	for p, a in zip(predicted, actual):
		sum += (p - a) * (p - a)
	sum /= len(predicted)
	return math.sqrt(sum)

def b0_coeff(dataset, b1):
	my = dataset.Y.mean()
	mx = dataset.X.mean()
	return my - b1 * mx

def b1_coeff(dataset):
	mx = dataset.X.mean()
	my = dataset.Y.mean()
	sum_nominator = 0
	sum_denominator = 0
	for index, row in dataset.iterrows():
		sum_nominator += (row['X'] - mx) * (row['Y'] - my)
		sum_denominator += (row['X'] - mx) * (row['X'] - mx)
	return sum_nominator / sum_denominator

def predict(predictors, b0, b1):
	ans = []
	for p in predictors:
		ans.append(b0 + b1 * p)
	return ans


def main():

	df = pd.read_csv('slr.csv')

	plt.scatter(df.X, df.Y, color='black')
	plt.xlabel('x')
	h = plt.ylabel('y')
	h.set_rotation(0)


	print("Shape:", df.shape)

	train_set = df.iloc[:40]
	test_set  = df.iloc[40:]

	b1 = b1_coeff(train_set)
	b0 = b0_coeff(train_set, b1)

	X = np.linspace(0, 120, 40)
	Y = list([b0 + b1 * x for x in X])
	plt.plot(X, Y, color='red')
	plt.show()

	test_predicted = predict(test_set.X, b0, b1)
	rm = rmse(test_predicted, test_set.Y)

	r2 = r2_score(test_predicted, test_set.Y)
	print("Corr: ", statistics.pearsonr(df.X, df.Y))
	print("Simple regression\nintercept: {}\nslope: {}\nrmse: {}\nr2: {}\n".format(b0, b1, rm, r2))

	regression_model = LinearRegression()
	regression_model.fit(np.array(train_set.X).reshape(-1, 1), np.array(train_set.Y).reshape(-1, 1))

	test_set_x = np.array(test_set.X).reshape(-1, 1)
	test_set_y = np.array(test_set.Y).reshape(-1, 1)
	b1_s = regression_model.coef_[0]
	b0_s = regression_model.intercept_

	# metrics

	predicted = regression_model.predict(test_set_x)

	r2_s = regression_model.score(test_set_y, predicted)
	mse = skm.mean_squared_error(test_set_y, predicted)
	skm_r2 = skm.r2_score(test_set_y, predicted)
	exp_var = skm.explained_variance_score(test_set_y, predicted)
	print("sklearn regression\nintercept: {}\nslope: {}\nr2 of model: {}\nr2 from metrics package: {}\nmse: {}\nexp_var: {}"
		  .format(b0_s, b1_s, r2_s, skm_r2, mse, exp_var))

if __name__ == '__main__':
	main()

