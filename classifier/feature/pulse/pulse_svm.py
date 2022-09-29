#   -*- coding:utf-8 -*-
#   The pulse_svm.py in PE_CLF
#   created by Jiang Feng(silencejiang@zju.edu.cn)
#   created at 3:29 on 2022/9/30

from utils.logger import Logger
from data_loader.feature_loader import FeatureLoader


log = Logger().get_log()
fl = FeatureLoader()
fl.load_data("0.17.0_mf_20220526_163827.csv")
X_train,X_test,y_train,y_test =fl.split()

from classifier.myclassifer import My_classifier as mclf
from sklearn.svm import LinearSVC, NuSVC,SVC

# clf = mclf(LinearSVC())
# clf.set_Datasets(X_train, y_train, X_test, y_test)
# clf.fit_predict()
# # unable to converge
# clf = mclf(NuSVC())
# clf.set_Datasets(X_train, y_train, X_test, y_test)
# clf.fit_predict()
#
# clf = mclf(SVC())
# clf.set_Datasets(X_train, y_train, X_test, y_test)
# clf.fit_predict()


# log.info("Linear")
# from sklearn.model_selection import GridSearchCV
# paras = [
#     {'penalty':['l1','l2'],
#      'loss':['hinge','squared_hinge'],
#      'multi_class':['ovr','crammer_singer']}
# ]
#
# lsvc = LinearSVC()
#
# grid_search = GridSearchCV(lsvc, paras,cv=3,scoring="roc_auc")
# grid_search.fit(X_train,y_train)
#
# log.info("最优参数")
# log.info(grid_search.best_params_)
# log.info("最优模型")
# log.info(grid_search.best_estimator_)
# log.info("具体表现")
# log.info(grid_search.cv_results_)
#
# clf = mclf(grid_search.best_estimator_)
# clf.set_Datasets(X_train, y_train, X_test, y_test)
# clf.fit_predict()

# log.info("NuSVC")
# from sklearn.model_selection import GridSearchCV
# paras = [
#     {'nu':[0.25,0.5,0.75],
#      'kernel':['linear','poly','rbf','sigmoid','precomputed'],
#      'decision_function_shape':['ovo','ovr']}
# ]
#
# nsvc = NuSVC()
#
# grid_search = GridSearchCV(nsvc, paras,cv=3,scoring="roc_auc")
# grid_search.fit(X_train,y_train)
#
# log.info("最优参数")
# log.info(grid_search.best_params_)
# log.info("最优模型")
# log.info(grid_search.best_estimator_)
# log.info("具体表现")
# log.info(grid_search.cv_results_)
#
# clf = mclf(grid_search.best_estimator_)
# clf.set_Datasets(X_train, y_train, X_test, y_test)
# clf.fit_predict()


log.info("C_SVC")
from sklearn.model_selection import GridSearchCV
paras = [
    {'C':[1,10,100],
     'kernel':['linear','poly','rbf','sigmoid','precomputed'],
     'decision_function_shape':['ovo','ovr']}
]

svc = SVC()

grid_search = GridSearchCV(svc, paras,cv=3,scoring="roc_auc")
grid_search.fit(X_train,y_train)

log.info("最优参数")
log.info(grid_search.best_params_)
log.info("最优模型")
log.info(grid_search.best_estimator_)
log.info("具体表现")
log.info(grid_search.cv_results_)

clf = mclf(grid_search.best_estimator_)
clf.set_Datasets(X_train, y_train, X_test, y_test)
