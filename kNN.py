import math
import statistics

from dataset_loader import cross_validation_split, accuracy, minmax_normalization

def euclidean_distance(v1, v2):
	res = 0
	for i in range(len(v1) - 1):
		res += (v1[i] - v2[i]) * (v1[i] - v2[i])
	return math.sqrt(res)

def get_neighbors(v, dataset, k):
	temp = {i : dataset[i] for i in range(len(dataset))}
	for i in temp:
		row = list(j for j in range(len(temp) - 1))
		temp[i] = euclidean_distance(v, row)
	# lambda item: item[1] means sorting by second value == value in (key, value)
	temp = dict(sorted(temp.items(), key=lambda item: item[1]))
	neighbors = []
	cnt = 0
	for i in temp:
		if cnt < k:
			neighbors.append(dataset[i])
		else: break
		cnt+=1
	return neighbors

def predict_classification(target, dataset, k, answer_column=-1):
	neighbors = get_neighbors(target, dataset, k)
	y_train = [row[answer_column] for row in neighbors]
	return max(set(y_train), key=y_train.count)

def predict_regression(target, dataset, k, answer_column=-1):
	neighbors = get_neighbors(target, dataset, k)
	y_train = [row[answer_column] for row in neighbors]
	mean = 0
	for y in y_train:
		mean += y
	return mean / k

def kNN_algorithm_mean_accuracy(dataset, folds_n, k, type='classification'):
	minmax_normalization(dataset)
	alg = predict_classification if type == 'classification' else predict_regression
	folds = cross_validation_split(dataset, folds_n)
	accuracies = []
	for fold in folds:
		train_set = list(folds)
		train_set.remove(fold)
		train_set = sum(train_set, [])
		predicted = []
		for row in fold:
			predicted.append(alg(row, train_set, k))
		accuracies.append(accuracy([row[-1] for row in fold], predicted))
	return statistics.mean(accuracies)

def kNN_algorithm(target, train_set, k, mapping=None, type='classification'):
	alg = predict_classification if type == 'classification' else predict_regression
	predict = alg(target, train_set, k)
	if mapping is None or len(mapping) == 0:
		return predict
	for key in mapping:
		if predict == mapping[key]:
			return key

def kNN_algorithm(target, train_set, k, mapping=None, type='classification'):
	alg = predict_classification if type == 'classification' else predict_regression
	predict = []
	for row in target:
		predict.append(alg(row, train_set, k))
	if mapping is None or len(mapping) == 0:
		return predict
	res = []
	for x in predict:
		for key in mapping:
			if x == mapping[key]:
				res.append(key)
	return res