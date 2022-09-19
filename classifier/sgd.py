#   -*- coding:utf-8 -*-
#   The sgd.py in PE_CLF
#   created by Jiang Feng(silencejiang@zju.edu.cn)
#   created at 4:05 on 2022/9/20

from utils.logger import Logger
from data_loader.feature_loader import FeatureLoader


log = Logger().get_log()
fl = FeatureLoader()
fl.load_data("0.17.0_mf_20220526_163827.csv")
X_train,X_test,y_train,y_test =fl.split()

from classifier.myclassifer import My_classifier as mclf
from sklearn.linear_model import SGDClassifier

clf = mclf(SGDClassifier(random_state=42))
clf.set_Datasets(X_train, y_train, X_test, y_test)
clf.fit_predict()



