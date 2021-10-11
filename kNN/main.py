from dataset_loader import *
from kNN import *

print('a')
book_1 = load_dataset_column_as_row('../md_features.csv')
book_2 = load_dataset_column_as_row('E:\\stud\\ml_algorithms\\portret_gogol_features.csv')
b = euclidian_distance_quality_feaures(book_1, book_2)
a = 5
