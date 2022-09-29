#   -*- coding:utf-8 -*-
#   The person_rf.py in PE_CLF
#   created by Jiang Feng(silencejiang@zju.edu.cn)
#   created at 5:34 on 2022/9/30

from utils.logger import Logger
from data_loader.feature_loader import FeatureLoader


log = Logger().get_log()
fl = FeatureLoader()
fl.load_data("0.17.0_mf_20220526_163827.csv")
X_train,X_test,y_train,y_test ,tat, tet=fl.split_by_person()

from classifier.myclassifer import My_classifier as mclf
from sklearn.ensemble import  RandomForestClassifier

clf = mclf(RandomForestClassifier(random_state=42))
clf.set_Datasets(X_train, y_train, X_test, y_test)
clf.fit_predict(tet)