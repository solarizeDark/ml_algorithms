import random
import statistics
from _csv import writer
from csv import reader

def load_dataset(filename, del_first=True):
	dataset = []
	with open(filename, 'r') as file:
		csv_reader = reader(file)
		if del_first:
			csv_reader.__next__()
		for row in csv_reader:
			if not row:
				continue
			dataset.append(row)
	return dataset

# special
def load_dataset_column_as_row(filename, del_first=True):
	features = {}
	with open(filename, 'r', encoding='utf-8') as file:
		csv_reader = reader(file)
		if del_first:
			csv_reader.__next__()
		for row in csv_reader:
			if not row:
				continue
			features[row[1]] = row[0]
	return features

def func_mapper(row):
	mapping = {}
	for i in range(len(row)):
		try:
			float(row[i].strip())
			mapping[i] = string_column_to_float
		except ValueError:
			mapping[i] = string_column_to_int
	return mapping

def string_column_to_float(dataset, column):
	for row in dataset:
		row[column] = float(row[column].strip())

# mapping between string value and int
def string_column_to_int(dataset, column):
	unique_values = set(row[column] for row in dataset)
	values = list(i for i in range(len(unique_values)))
	mapping = dict.fromkeys(unique_values)
	cnt = 0
	for i in mapping:
		mapping[i] = values[cnt]
		cnt+=1
	for row in dataset:
		row[column] = mapping[row[column]]
	return mapping

def dataset_values_converter(dataset, mapping):
	str_int_mappings = []
	for column in range(len(dataset[0])):
		str_int_mappings.append(mapping[column](dataset, column))
	return [x for x in str_int_mappings if x is not None]

def convert_to_numeral(dataset):
	func_mapping = func_mapper(dataset[0])
	string_int_mapping = dataset_values_converter(dataset, func_mapping)
	return dataset, string_int_mapping

def delete_heading(dataset):
	return list(dataset[i] for i in range(1, len(dataset)))

def delete_column(dataset, indexes):
	new_dataset = []
	for row in dataset:
		new_row = list(row)
		for index in indexes:
			new_row.remove(row[index])
		new_dataset.append(new_row)
	return new_dataset

def train_test_split(dataset, per=0.6):
	train_set = []
	train_size = int(len(dataset) * 0.6)
	dataset_copy = list(dataset)
	for x in range(train_size):
		index = random.randrange(len(dataset_copy))
		train_set.append(dataset_copy[index])
		dataset_copy.pop(index)
	return train_set, dataset_copy

def cross_validation_split(dataset, folds_n=5):
	folds = []
	fold_size = int(len(dataset) / folds_n)
	dataset_copy = list(dataset)
	for x in range(folds_n):
		fold = []
		for i in range(fold_size):
			index = random.randrange(len(dataset_copy))
			fold.append(dataset_copy.pop(index))
		folds.append(fold)
	return folds

# columns - which should be normalized
def minmax_normalization(dataset, columns):
	minmax = []
	for i in columns:
		column = [row[i] for row in dataset]
		minmax.append([min(column), max(column)])
	for j in range(len(dataset)):
		for i in range(len(dataset[0])):
			dataset[j][i] = (dataset[j][i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])

def null_handler(dataset, column):
	average = list(row[column] for row in dataset if row[column] and row[column] != '0')
	for i in range(len(average)):
		average[i] = float(average[i])
	mean = round(statistics.mean(average), 2)
	for i in range(len(dataset)):
		for j in range(len(dataset[i])):
			if dataset[i][j] == '':
				dataset[i][j] = str(mean)

def change_columns(dataset, col_i, col_j):
	for row in dataset:
		temp = row[col_i]
		row[col_i] = row[col_j]
		row[col_j] = temp

def accuracy(actual, predicted):
	cnt = 0
	for i in range(len(actual)):
		if actual[i] == predicted[i]:
			cnt+=1
	return cnt / float(len(actual))

def dataset_writer(heading, dataset, filename):
	with open(filename, 'w', newline='') as file:
		csv_writer = writer(file)
		csv_writer.writerow(heading)
		for row in dataset:
			csv_writer.writerow(row)
