#   -*- coding:utf-8 -*-
#   The lr.py in PE_CLF
#   created by Jiang Feng(silencejiang@zju.edu.cn)
#   created at 5:25 on 2022/9/20

from utils.logger import Logger
from data_loader.feature_loader import FeatureLoader


log = Logger().get_log()
fl = FeatureLoader()
fl.load_data("0.17.0_mf_20220526_163827.csv")
X_train,X_test,y_train,y_test =fl.split()

from classifier.myclassifer import My_classifier as mclf
from sklearn.linear_model import LogisticRegression

clf = mclf(LogisticRegression(solver= "liblinear", random_state= 42,max_iter= 3000))
clf.set_Datasets(X_train, y_train, X_test, y_test)
clf.fit_predict()




from sklearn.model_selection import GridSearchCV
paras = [
    {'penalty':['l2','l1','elasticnet','none'],
     'solver':['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
    'C':[1,10,100,100]}
]

lr = LogisticRegression(random_state= 42,max_iter= 30000)

grid_search = GridSearchCV(lr, paras,cv=3,scoring="roc_auc")
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