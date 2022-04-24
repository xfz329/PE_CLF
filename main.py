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


from sklearn.linear_model import SGDClassifier

sgd_clf = SGDClassifier(random_state= 42)
sgd_clf.fit(X_train , y_train)

from sklearn.model_selection import  cross_val_score
scores = cross_val_score(sgd_clf, X_train, y_train, cv=5, scoring="accuracy")
log.info(scores)

from sklearn.model_selection import  cross_val_predict

y_train_pred = cross_val_predict(sgd_clf, X_train,y_train , cv =5)

from sklearn.metrics import confusion_matrix,precision_score,recall_score,f1_score

cm = confusion_matrix(y_train,y_train_pred)
log.info(cm)
prec = precision_score(y_train,y_train_pred)
log.info(prec)
recall = recall_score(y_train,y_train_pred)
log.info(recall)
f1 = f1_score(y_train, y_train_pred)
log.info(f1)


y_train_scores = cross_val_predict(sgd_clf, X_train,y_train , cv =5, method= "decision_function")
log.info(y_train_scores.shape)
log.info(y_train_scores)

from sklearn.metrics import precision_recall_curve

precisions, recalls, thresholds = precision_recall_curve(y_train, y_train_scores)
log.info("precisions recalls thresholds")
log.info(precisions)
log.info(recalls)
log.info(thresholds)

import  matplotlib.pyplot as plt
def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision", linewidth=2)
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall", linewidth=2)
    plt.xlabel("Threshold", fontsize=16)
    plt.legend(loc="upper left", fontsize=16)
    plt.ylim([0, 1])

plt.figure(figsize=(8, 4))
plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
plt.xlim([-6000000, 15000000])
# save_fig("precision_recall_vs_threshold_plot")
plt.show()


from sklearn.metrics import  roc_curve,roc_auc_score

fpr , tpr , thresholds = roc_curve(y_train, y_train_scores)
fs.plot_roc_curve(fpr, tpr, "roc_of_sgd")

log.info(roc_auc_score(y_train, y_train_scores))


y_sgd_pred = sgd_clf.predict(X_test)

from sklearn.metrics import  accuracy_score
log.info(accuracy_score(y_test,y_sgd_pred))

cm = confusion_matrix(y_test,y_sgd_pred)
log.info(cm)


from  sklearn.model_selection import  GridSearchCV

param_grid = [
    {"loss":["hinge","modified_huber","log"], "penalty":["l2","l1","elasticnet"]}
]
grid_search = GridSearchCV(sgd_clf,param_grid,cv = 5)
grid_search.fit(X_train,y_train)
log.info(grid_search.best_estimator_)

sgd_clf = grid_search.best_estimator_
y_sgd_pred = sgd_clf.predict(X_test)

from sklearn.metrics import  accuracy_score
log.info(accuracy_score(y_test,y_sgd_pred))

cm = confusion_matrix(y_test,y_sgd_pred)
log.info(cm)


# from sklearn.tree import  DecisionTreeClassifier
#
# tree_clf = DecisionTreeClassifier(random_state = 42)
# tree_clf.fit(data_prepared, y)
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

