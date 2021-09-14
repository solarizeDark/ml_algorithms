from dataset_loader import *
from kNN import kNN_algorithm_mean_accuracy, kNN_algorithm

train_set = load_dataset('train.csv')
train_set = delete_column(train_set, [0, 3, 8, 10, 11])
null_handler(train_set, 4)
change_columns(train_set, 1, -1)
train_set, mapping = convert_to_numeral(train_set)
minmax_normalization(train_set)

test_set = load_dataset('test.csv')
test_set = delete_column(test_set, [0, 2, 7, 9, 10])
null_handler(test_set, 4)
test_set, mapping_test = convert_to_numeral(test_set)
minmax_normalization(test_set)

for i in range(len(train_set[0]) - 1):
    change_columns(train_set, i, i + 1)

for i in range(5, 15, 2):
    print(i, kNN_algorithm_mean_accuracy(train_set, 5, i))

predict = kNN_algorithm(test_set, train_set, 5)
answer = []
for i, j in zip(range(892, 1310), range(len(predict))):
    answer.append([i, int(predict[j])])
dataset_writer(['PassengerId','Survived'], answer, 'titanic_predict.csv')

# iris = load_dataset('iris.csv')
# train_set, mapping = convert_to_numeral(iris)
# minmax_normalization(iris)
# for i in range(5, 25, 2):
#     print(i, kNN_algorithm_mean_accuracy(train_set, 5, i))

