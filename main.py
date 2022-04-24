#   -*- coding:utf-8 -*-
#   The main.py.py in PE_CLF
#   created by Jiang Feng(silencejiang@zju.edu.cn)
#   created at 11:03 on 2022/4/24
from utils.figuresaver import FigureSaver
from utils.logger import Logger
from data.data import Data
import numpy as np

log = Logger("clf").get_log()
d = Data()
fs = FigureSaver()
d.load_data("total.csv")
data= d.drop_invalid_value()
print(data.columns)


from sklearn.model_selection import  StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1 ,test_size= 0.2 , random_state= 42)
for train_index, test_index in split.split(data, data["PE_state"]):
    train_set = data.loc[train_index]
    test_set = data.loc[test_index]

corr_matrix = data.corr()
log.info(corr_matrix["PE_state"].sort_values(ascending= False))
X_train = train_set.drop("PE_state", axis = 1)
y_train = train_set["PE_state"].copy()
X_test = test_set.drop("PE_state", axis = 1)
y_test = test_set["PE_state"].copy()
X_train.info()
y_train.info()

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import  StandardScaler
#
# pipeline = Pipeline([
#     ('')
#
# ])

std_scaler = StandardScaler()
data_prepared = std_scaler.fit_transform(X_train)

from classifier.myclassifer import My_classifier as mclf
from sklearn.linear_model import SGDClassifier

m = mclf(SGDClassifier(random_state=42))
m.set_Datasets(X_train,y_train,X_test,y_test)
m.fit()
m.cross_val_score()
m.cross_val_predict()
m.performance_judge()
m.train_scores(**{"cv":5})
m.precision_recall_curve()
m.roc_curve()
m.predict()
m.performance_judge(False)

param_grid = [
    {"loss":["hinge","modified_huber","log"], "penalty":["l2","l1","elasticnet"]}
]
m.grid_search(**{"param_grid":param_grid, "cv":5})

from sklearn.tree import  DecisionTreeClassifier


mtree = mclf(DecisionTreeClassifier(random_state = 42))
mtree.set_Datasets(X_train,y_train,X_test,y_test)
mtree.fit()
mtree.cross_val_score()
mtree.cross_val_predict()
mtree.performance_judge()
mtree.train_scores(**{"cv":5})
mtree.precision_recall_curve()
mtree.roc_curve()
mtree.predict()
mtree.performance_judge(False)

#
# from sklearn.metrics import  mean_squared_error
# predictions = tree_clf.predict(data_prepared)
# tree_mse = mean_squared_error(y, predictions)
# tree_rmse = np.sqrt(tree_mse)
# log.info(tree_rmse)
#
#
# from sklearn.model_selection import  cross_val_score
#
# scores = cross_val_score(tree_clf, data_prepared, y, scoring= "neg_mean_squared_error",cv = 10)
# rmse_scores = np.sqrt(-scores)
#
#
# def display_scores(scores):
#     log.info("Scores:" + str(scores))
#     log.info("Mean:" + str(scores.mean()))
#     log.info("Standard deviation:" + str(scores.std()))
#
# display_scores(rmse_scores)


log.info("over")

