#   -*- coding:utf-8 -*-
#   The pulse_knn2.py in PE_CLF
#   created by Jiang Feng(silencejiang@zju.edu.cn)
#   created at 3:52 on 2022/10/13
from utils.logger import Logger
from data_loader.point_loader2 import PointLoader2

log = Logger().get_log()
pl = PointLoader2()
pl.load_data("0.17.0_mp2_20221013_004934.csv")
X_train,X_test,y_train,y_test =pl.split()

from classifier.myclassifer import My_classifier as mclf
from sklearn.neighbors import KNeighborsClassifier

clf = mclf(KNeighborsClassifier())
clf.set_Datasets(X_train, y_train, X_test, y_test)
clf.fit_predict()


from sklearn.model_selection import GridSearchCV
paras = [
    {   'n_neighbors':[3,5,7,9],
        'weights':['uniform','distance']}
]

knn = KNeighborsClassifier()

grid_search = GridSearchCV(knn, paras,cv=3,scoring="roc_auc")
grid_search.fit(X_train,y_train)

log.info("最优参数")
log.info(grid_search.best_params_)
log.info("最优模型")
log.info(grid_search.best_estimator_)
log.info("具体表现")
log.info(grid_search.cv_results_)

clf = mclf(grid_search.best_estimator_)
clf.set_Datasets(X_train, y_train, X_test, y_test)
clf.fit_predict()