#   -*- coding:utf-8 -*-
#   The pulse_rf.py in PE_CLF
#   created by Jiang Feng(silencejiang@zju.edu.cn)
#   created at 3:33 on 2022/9/30

from utils.logger import Logger
from data_loader.feature_loader import FeatureLoader


log = Logger().get_log()
fl = FeatureLoader()
fl.load_data("0.17.0_mf_20220526_163827.csv")
X_train,X_test,y_train,y_test =fl.split()

from classifier.myclassifer import My_classifier as mclf
from sklearn.ensemble import  RandomForestClassifier

clf = mclf(RandomForestClassifier(random_state=42))
clf.set_Datasets(X_train, y_train, X_test, y_test)
clf.fit_predict()

p50=[
'CVALF_9','LVRF_9','SVD_10','SVAF_10','LVALF_7','SVAT_10','CVRF_11',
    'SVAR_10','STDZ_3','LVRF_8','LVALF_6','CVD_11','STDZ_1','CVALF_8',
    'SVD_9','SVAAR_10','CVALF_1','SVAAR_8','SVD_8','SVAR_9','SVAF_9',
    'CVAAF_1','LVLF_8','SVAT_8','CVRF_1','CVAAF_2','SVAR_8','SVAF_2',
    'CVRR_11','LVRF_1','SVSR_7','CVRF_3','CVALR_10','CVALF_7']
p70=['CVALF_9','LVRF_9','SVD_10','SVAF_10','LVALF_7','SVAT_10','CVRF_11',
     'SVAR_10','STDZ_3','LVRF_8','LVALF_6','CVD_11','STDZ_1','CVALF_8',
     'SVD_9','SVAAR_10','CVALF_1','SVAAR_8','SVD_8','SVAR_9','SVAF_9',
     'CVAAF_1','LVLF_8','SVAT_8','CVRF_1','CVAAF_2','SVAR_8','SVAF_2',
     'CVRR_11','LVRF_1','SVSR_7','CVRF_3','CVALR_10','CVALF_7','CVAAR_5',
     'LVRF_2','LVALF_3','CVRF_4','CVALF_2','CVALF_4','CVALF_6','CVALF_10',
     'CVAAR_8','LVLF_1','LVD_1','LVALF_1','LVALF_5','SVAT_1','SVAT_2','SVD_3',
     'SVSR_9','SVSR_10','CVRR_8','CVRR_9','CVRR_10','CVRF_2','CVD_1','CVALF_5',
     'CVAAR_9','CVAAF_3','LVLF_7','LVRF_7','LVD_2','LVALF_2','SVAR_1','SVAF_1',
     'SVAF_8','SVD_2','SVSR_5','SVAAR_9','CVD_2','CVD_3','CVALR_7','CVALF_3']

import pandas as pd

X_train_50 = pd.DataFrame(X_train,columns=p50)
X_test_50 = pd.DataFrame(X_test,columns=p50)
# print(X_train_50)
# print(X_test_50)
# print(y_train)
# print(y_test)

# clf = mclf(RandomForestClassifier(random_state=42))
# clf.set_Datasets(X_train_50, y_train,X_test_50, y_test)
# clf.fit_predict()
#
# X_train_70 = pd.DataFrame(X_train,columns=p70)
# X_test_70 = pd.DataFrame(X_test,columns=p70)
#
# clf = mclf(RandomForestClassifier(random_state=42))
# clf.set_Datasets(X_train_70, y_train,X_test_70, y_test)
# clf.fit_predict()



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