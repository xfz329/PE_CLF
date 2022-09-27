#   -*- coding:utf-8 -*-
#   The tree.py in PE_CLF
#   created by Jiang Feng(silencejiang@zju.edu.cn)
#   created at 5:02 on 2022/9/20

from utils.logger import Logger
from data_loader.feature_loader import FeatureLoader


log = Logger().get_log()
fl = FeatureLoader()
fl.load_data("0.17.0_mf_20220526_163827.csv")
X_train,X_test,y_train,y_test =fl.split()

from classifier.myclassifer import My_classifier as mclf
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

clf = mclf(DecisionTreeClassifier(random_state=42,max_depth=3))
clf.set_Datasets(X_train, y_train, X_test, y_test)
clf.fit_predict()

from sklearn.model_selection import GridSearchCV
paras = [
    {'criterion':["gini",'entropy','log_loss'],
     'splitter':['best','random'],
     'max_depth':[3,4,5],
     'max_features':['sqrt','log2',None]}
]

tr = DecisionTreeClassifier(random_state=42)

grid_search = GridSearchCV(tr, paras,cv=3,scoring="roc_auc")
grid_search.fit(X_train,y_train)

log.info("最优参数")
log.info(grid_search.best_params_)
log.info("最优模型")
log.info(grid_search.best_estimator_)
log.info("具体表现")
log.info(grid_search.cv_results_)

import os
from sklearn.tree import export_graphviz,DecisionTreeClassifier
import pydotplus
from utils.project_dir import ProjectDir
out_dir =  ProjectDir().dir_figures
clf = grid_search.best_estimator_


out_file = os.path.join(out_dir,"dt_clf"+str(clf)+".dot")
export_graphviz(
    clf,
    out_file=out_file,
    feature_names=X_train.columns,
    class_names=["Health","PE"],
    rounded=True,
    filled=True
)
graph = pydotplus.graph_from_dot_file(out_file)
# Image(graph.create_png())
out_file = os.path.join(out_dir, "dt_clf" + str(clf) + ".png")
graph.write_png(out_file)

clf = mclf(grid_search.best_estimator_)
clf.set_Datasets(X_train, y_train, X_test, y_test)
clf.fit_predict()