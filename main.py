#   -*- coding:utf-8 -*-
#   The main.py.py in PE_CLF
#   created by Jiang Feng(silencejiang@zju.edu.cn)
#   created at 11:03 on 2022/4/24
from utils.figuresaver import FigureSaver
from utils.logger import Logger
from data_loader.feature_loader import FeatureLoader

log = Logger().get_log()
fl = FeatureLoader()
fl.load_data()
X_train,X_test,y_train,y_test =fl.split()
fl.corr()

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
# clfs = [mtree]

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

