#   -*- coding:utf-8 -*-
#   The rf.py in PE_CLF
#   created by Jiang Feng(silencejiang@zju.edu.cn)
#   created at 7:42 on 2022/9/21

from utils.logger import Logger
from data_loader.feature_loader import FeatureLoader


log = Logger().get_log()
fl = FeatureLoader()
fl.load_data("0.17.0_mf_20220526_163827.csv")
X_train,X_test,y_train,y_test =fl.split()

from classifier.myclassifer import My_classifier as mclf
from sklearn.ensemble import  RandomForestClassifier

# clf = mclf(RandomForestClassifier(random_state=42,n_jobs= -1, n_estimators= 500))
# clf.set_Datasets(X_train, y_train, X_test, y_test)
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

# best
clf = mclf(RandomForestClassifier(criterion='entropy', max_features=None, n_estimators=500, random_state=42, verbose= 1,n_jobs= -1))
# clf = mclf(RandomForestClassifier(random_state=42,n_jobs= -1, n_estimators= 500))
clf.set_Datasets(X_train, y_train, X_test, y_test)
clf.fit_predict()

# import joblib
#
# clf = mclf(0)
# clf.set_clf(r"D:\UrgeData\Documents\Codes\Graduate\PE_CLF\models\20220921\model_RandomForestClassifier_20220921_104918.pkl")
# clf.set_Datasets(X_train, y_train, X_test, y_test)
# clf.predict_only()