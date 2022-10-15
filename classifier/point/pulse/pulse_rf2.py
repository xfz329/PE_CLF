#   -*- coding:utf-8 -*-
#   The pulse_rf2.py in PE_CLF
#   created by Jiang Feng(silencejiang@zju.edu.cn)
#   created at 3:55 on 2022/10/13
from utils.logger import Logger
from data_loader.point_loader2 import PointLoader2


log = Logger().get_log()
pl = PointLoader2()
pl.load_data("0.17.0_mp2_20221013_004934.csv")
X_train,X_test,y_train,y_test =pl.split()

from classifier.myclassifer import My_classifier as mclf
from sklearn.ensemble import  RandomForestClassifier

clf = mclf(RandomForestClassifier(random_state=42))
clf.set_Datasets(X_train, y_train, X_test, y_test)
clf.fit_predict(show_imp=True)