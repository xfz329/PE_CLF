#   -*- coding:utf-8 -*-
#   The pulse_rf.py in PE_CLF
#   created by Jiang Feng(silencejiang@zju.edu.cn)
#   created at 3:33 on 2022/9/30
import pandas as pd

from utils.logger import Logger
from data_loader.feature_loader import FeatureLoader
from classifier.feature.pulse.pulse_rf_sort import  RandomForestPercentage


log = Logger().get_log()
fl = FeatureLoader()
fl.load_data("0.17.0_mf_20220526_163827.csv")
X_train,X_test,y_train,y_test =fl.split()

from classifier.myclassifer import My_classifier as mclf
from sklearn.ensemble import  RandomForestClassifier

log.info("percentage is 100%")
clf = mclf(RandomForestClassifier(random_state=42))
clf.set_Datasets(X_train, y_train, X_test, y_test)
clf.fit_predict()


rtf = RandomForestPercentage()
for x in range(90,9,-10):
    log.info("percentage is "+ str(x)+"%")
    log.info("feature num is "+str(len(rtf.get(x))))
    X_train_x = pd.DataFrame(X_train,columns=rtf.get(x))
    X_test_x = pd.DataFrame(X_test,columns=rtf.get(x))
    clf = mclf(RandomForestClassifier(random_state=42))
    clf.set_Datasets(X_train_x, y_train,X_test_x, y_test)
    clf.fit_predict()


# from sklearn.model_selection import GridSearchCV
# paras = [
#     {   'criterion':['gini','entropy','log_loss'],
#         'n_estimators':[100,300,500],
#         'max_depth':[3,4,5,6,None],
#         'max_features':['sqrt','log2',None],
#     }
# ]
#
# rf = RandomForestClassifier(random_state=42,n_jobs= -1,verbose= 1)
#
# grid_search = GridSearchCV(rf, paras,cv=3,scoring="roc_auc")
# log.info("GridSearch Begins")
# log.info(paras)
# grid_search.fit(X_train,y_train)
#
# log.info("最优参数")
# log.info(grid_search.best_params_)
# log.info("最优模型")
# log.info(grid_search.best_estimator_)
# log.info("具体表现")
# log.info(grid_search.cv_results_)
#
# # best
# clf = mclf(RandomForestClassifier(criterion='entropy', max_features=None, n_estimators=500, random_state=42, verbose= 1,n_jobs= -1))
# # clf = mclf(RandomForestClassifier(random_state=42,n_jobs= -1, n_estimators= 500))
# clf.set_Datasets(X_train, y_train, X_test, y_test)
# clf.fit_predict()