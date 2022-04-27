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

from sklearn.preprocessing import  StandardScaler
std_scaler = StandardScaler()

corr_matrix = data.corr()
log.info(corr_matrix["PE_state"].sort_values(ascending= False))
X_train = train_set.drop("PE_state", axis = 1)
X_train = std_scaler.fit_transform(X_train)
y_train = train_set["PE_state"].copy()
X_test = test_set.drop("PE_state", axis = 1)
X_test = std_scaler.fit_transform(X_test)
y_test = test_set["PE_state"].copy()
# X_train.info()
# y_train.info()

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
from sklearn.tree import  DecisionTreeClassifier
from sklearn.ensemble import  RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
# ComplementNB 不能接受数据特征中存在负值
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import  SVC
import  time


msgd = mclf(SGDClassifier(random_state=42))
mtree = mclf(DecisionTreeClassifier(random_state = 42, max_depth=3))
# mtree2 = mclf(DecisionTreeClassifier(random_state = 42, max_depth=4))
mrf = mclf(RandomForestClassifier(n_jobs= -1, n_estimators= 500,max_leaf_nodes=16))
mknn = mclf(KNeighborsClassifier())
mgb = mclf(GaussianNB())
mlog= mclf(LogisticRegression(solver= "liblinear", random_state= 42,max_iter= 3000))
mlp = mclf(MLPClassifier(random_state=1, max_iter=300))
msvm =mclf(LinearSVC(C=1,loss= "hinge",max_iter=100000))
clfs = [msgd,mtree,mrf,mknn,mgb,mlog,mlp,msvm]
# clfs = [mtree,mtree2]

for clf in clfs:
    start = time.time()
    clf.set_Datasets(X_train, y_train, X_test, y_test)
    clf.fit()
    clf.performance_judge()
    clf.predict()
    clf.performance_judge(on_test_set=True)
    end = time.time()
    log.info("time consuming "+ str(end-start))

# param_grid = [
#     {"loss":["hinge","modified_huber","log"], "penalty":["l2","l1","elasticnet"]}
# ]
# m.grid_search(**{"param_grid":param_grid, "cv":5})

log.info("over")

