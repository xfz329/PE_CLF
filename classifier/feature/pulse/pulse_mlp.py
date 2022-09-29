#   -*- coding:utf-8 -*-
#   The pulse_mlp.py in PE_CLF
#   created by Jiang Feng(silencejiang@zju.edu.cn)
#   created at 3:23 on 2022/9/30
from utils.logger import Logger
from data_loader.feature_loader import FeatureLoader


log = Logger().get_log()
fl = FeatureLoader()
fl.load_data("0.17.0_mf_20220526_163827.csv")
X_train,X_test,y_train,y_test =fl.split()

from classifier.myclassifer import My_classifier as mclf
from sklearn.neural_network import MLPClassifier

clf = mclf(MLPClassifier(max_iter=300))
clf.set_Datasets(X_train, y_train, X_test, y_test)
clf.fit_predict()