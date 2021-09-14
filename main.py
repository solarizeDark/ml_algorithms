from dataset_loader import *
from kNN import kNN_algorithm_mean_accuracy, kNN_algorithm

train_set = load_dataset('train.csv')
train_set = delete_column(train_set, [3, 8, 10, 11])
null_handler(train_set, 4)
change_columns(train_set, 1, -1)
train_set, mapping = convert_to_numeral(train_set)

test_set = load_dataset('test.csv')
test_set = delete_column(test_set, [2, 7, 9, 10])
null_handler(test_set, 4)
test_set, mapping_test = convert_to_numeral(test_set)
for i in range(1, 5):
    change_columns(test_set, i, -1)

for i in range(5, 30, 2):
    print(i, kNN_algorithm_mean_accuracy(train_set, 5, i))
predict = kNN_algorithm(test_set, train_set, 13)
answer = []
for i, j in zip(range(892, 1310), range(len(predict))):
    answer.append([i, int(predict[j])])
dataset_writer(['PassengerId','Survived'], answer, 'titanic_predict.csv')




