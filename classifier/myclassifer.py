#   -*- coding:utf-8 -*-
#   The myclassifer.py in PE_CLF
#   created by Jiang Feng(silencejiang@zju.edu.cn)
#   created at 21:11 on 2022/4/24
import os
import joblib
import time

import  matplotlib.pyplot as plt
import pandas as pd

from utils.logger import Logger
from utils.project_dir import ProjectDir
from utils.time_stamp import Time_stamp

class My_classifier:
    def __init__(self, clf,**kwargs):
        self.out_dir = ProjectDir().dir_figures
        self.model_dir = ProjectDir().dir_models
        self.data_dir = ProjectDir().dir_output
        self.clf = clf
        self.log = Logger("clf").get_log()
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.y_train_scores = None
        self.y_train_pred = None
        self.y_test_pred = None


    def info(self,pre,sub):
        self.log.info(pre+" of classifier " + str(self.clf)+" is\n"+str(sub))

    def set_clf(self, pkl_file):
        self.clf = joblib.load(pkl_file)
        ps = pkl_file.index(".pkl")
        txt_file = pkl_file[0:ps]+'.txt'
        with open(txt_file,'r') as f:
            content = f.read()
            self.log.info("model paras "+content)

    def set_Datasets(self,X_train,y_train,X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def fit(self):
        self.clf.fit(self.X_train,self.y_train)

    def predict(self):
        self.y_test_pred = self.clf.predict(self.X_test)
        self.log.info("predict detail is ")
        self.log.info(self.y_test_pred)
        t = Time_stamp()
        full_path = os.path.join(self.data_dir, "predict",t.get_day())
        if not os.path.exists(full_path):
            os.makedirs(full_path)
        file_path = os.path.join(full_path, "predict_" + t.get_time_stamp() + ".csv")
        self.y_test_pred.tofile(file_path,sep=',',format='%d')
        self.log.info("y_test_predict has been stored to the file " + file_path)


    def cross_val_score(self, **kwargs):
        from sklearn.model_selection import cross_val_score
        scores = cross_val_score(self.clf, self.X_train, self.y_train, **kwargs)
        self.info("cross_val_score",scores)

    def cross_val_predict(self,**kwargs):
        from sklearn.model_selection import cross_val_predict
        self.y_train_pred = cross_val_predict(self.clf, self.X_train, self.y_train,**kwargs)
        self.info("cross_val_predict",self.y_train_pred)

    def performance_judge(self,on_test_set = False):
        if not on_test_set:
            self.log.info("on train set")
            self.cross_val_score()
            self.cross_val_predict()
            self.get_performance(self.y_train,self.y_train_pred)
            self.train_scores(**{"cv":5})
            self.precision_recall_curve()
            self.roc_curve()
        else:
            self.y_test_pred = self.clf.predict(self.X_test)
            self.log.info("on test set")
            self.get_performance(self.y_test,self.y_test_pred)
            self.export2dot()

    def get_performance(self,y_true,y_pred):
        from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score,accuracy_score

        cm = confusion_matrix(y_true, y_pred)
        self.info("cm",cm)
        prec = precision_score(y_true, y_pred)
        self.info("precision_score",prec)
        recall = recall_score(y_true, y_pred)
        self.info("recall",recall)
        f1 = f1_score(y_true, y_pred)
        self.info("f1",f1)
        accuracy=accuracy_score(y_true, y_pred)
        self.info("accuracy",accuracy)

    def train_scores(self,**kwargs):
        from sklearn.model_selection import cross_val_predict
        if hasattr(self.clf,"decision_function"):
            y_train_scores = cross_val_predict(self.clf, self.X_train, self.y_train, method="decision_function",**kwargs)
        else:
            y_proba_scores = cross_val_predict(self.clf, self.X_train, self.y_train, method="predict_proba",**kwargs)
            y_train_scores = y_proba_scores[:,1]
        self.info("y_train_scores shape",y_train_scores.shape)
        self.info("y_train_scores",y_train_scores)
        self.y_train_scores = y_train_scores

    def precision_recall_curve(self):
        from sklearn.metrics import precision_recall_curve

        precisions, recalls, thresholds = precision_recall_curve(self.y_train, self.y_train_scores)
        self.info("precisions",precisions)
        self.info("recalls",recalls)
        self.info("thresholds",thresholds)
        self.plot_precision_recall_vs_threshold(precisions,recalls,thresholds)

    def plot_precision_recall_vs_threshold(self,precisions, recalls, thresholds):
        plt.figure(figsize=(8, 4))
        plt.plot(thresholds, precisions[:-1], "b--", label="Precision", linewidth=2)
        plt.plot(thresholds, recalls[:-1], "g-", label="Recall", linewidth=2)
        plt.xlabel("Threshold", fontsize=16)
        plt.legend(loc="upper left", fontsize=16)
        plt.ylim([0, 1])

        x_min = thresholds[1]
        x_max = thresholds[-1]

        plt.xlim([x_min, x_max])
        plt.title(self.clf.__class__.__name__)
        self.save_fig("Precision_recall_vs_threshold " + self.clf.__class__.__name__)
        # plt.show()

    def roc_curve(self):
        from sklearn.metrics import roc_curve, roc_auc_score

        fpr, tpr, thresholds = roc_curve(self.y_train, self.y_train_scores)
        auc = roc_auc_score(self.y_train,self.y_train_scores)
        self.plot_roc_curve(fpr, tpr, auc)
        self.info("auc",auc)

    def plot_roc_curve(self, fpr, tpr, auc, label=None):
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, linewidth=2, label=label)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.axis([0, 1, 0, 1])
        plt.xlabel('False Positive Rate', fontsize=16)
        plt.ylabel('True Positive Rate', fontsize=16)
        plt.title("AUC of "+self.clf.__class__.__name__+" is "+str(auc))
        self.save_fig("ROC of "+self.clf.__class__.__name__)
        # plt.show()

    def save_fig(self, fig_id, tight_layout=True, fig_extension="png", resolution=300):
        import os
        path = os.path.join(self.out_dir, fig_id + "." + fig_extension)
        self.log.info("Saving figure "+fig_id)
        if tight_layout:
            plt.tight_layout()
        plt.savefig(path, format=fig_extension, dpi=resolution)

    def grid_search(self,**kwargs):
        from  sklearn.model_selection import GridSearchCV
        gs = GridSearchCV(self.clf, **kwargs)
        gs.fit(self.X_train, self.y_train)
        self.log.info(gs.best_estimator_)

    def export2dot(self):
        from sklearn.tree import export_graphviz,DecisionTreeClassifier
        import pydotplus

        if isinstance(self.clf,DecisionTreeClassifier):
            self.log.info(self.clf)
            out_file = os.path.join(self.out_dir,"dt_clf"+str(self.clf)+".dot")
            export_graphviz(
                self.clf,
                out_file=out_file,
                feature_names=self.X_train.columns,
                class_names=["Health","PE"],
                rounded=True,
                filled=True
            )
            graph = pydotplus.graph_from_dot_file(out_file)
            # Image(graph.create_png())
            out_file = os.path.join(self.out_dir, "dt_clf" + str(self.clf) + ".png")
            graph.write_png(out_file)

    def show_importance(self, show = False):
        from sklearn.ensemble import RandomForestClassifier
        if show:
            import numpy as np

            # self.log.info("list detailed importance")
            # if isinstance(self.clf, RandomForestClassifier):
            #     for feat, importance in zip(self.X_train.columns, self.clf.feature_importances_):
            #         self.log.info('feature: {f}, importance: {i}'.format(f=feat, i=importance))
            #
            # # important_features = []
            # # for x, i in enumerate(self.clf.feature_importances_):
            # #     if i > np.average(self.feature_importances_):
            # #         important_features.append(x)
            # # important_names = self.X_train.columns[important_features > np.mean(important_features)]

            importances = pd.DataFrame({'feature': self.X_train.columns, 'importance': np.round(self.clf.feature_importances_, 3)})
            full_path = os.path.join(self.data_dir, "rf", Time_stamp().get_day())
            if not os.path.exists(full_path):
                os.makedirs(full_path)
            file_path = os.path.join(full_path, str(self.clf) + Time_stamp().get_time_stamp() + "_.csv")
            importances.to_csv(file_path)
            self.log.info("importance saved to file " + file_path)


    def fit_predict(self,true_label= None, show_imp = False):
        start = time.time()
        self.fit()
        self.performance_judge()
        self.predict()
        self.performance_judge(on_test_set=True)
        end = time.time()
        self.dump()
        self.log.info("time consuming " + str(end - start))
        self.show_importance(show_imp)
        if true_label is None:
            pass
        else:
            self.count_percentage(true_label)


    def predict_only(self):
        start = time.time()
        self.predict()
        self.performance_judge(on_test_set=True)
        end = time.time()
        self.log.info("time consuming " + str(end - start))

    def dump(self):
        t= Time_stamp()
        full_path = os.path.join(self.model_dir,t.get_day())
        if not os.path.exists(full_path):
            os.makedirs(full_path)
        self.log.info(full_path)
        ps = str(self.clf).index("(")
        pe = str(self.clf).index(")")
        clf_type = str(self.clf)[0:ps]
        ts = t.get_time_stamp()
        name = "model_"+clf_type+"_"+ts
        suffix = ".pkl"
        pkl_file = os.path.join(full_path,name+suffix)
        joblib.dump(self.clf,pkl_file)
        self.log.info(pkl_file)

        clf_conf = str(self.clf)[ps+1:pe]
        self.log.info(clf_conf)
        suffix = ".txt"
        txt_file = os.path.join(full_path,name+suffix)
        with open(txt_file,"w") as f:
            f.write(clf_conf)

    def count_percentage(self,true_label):
        person_list = true_label["person_name"].values.tolist()
        pure_list = list(set(person_list))
        predict = self.y_test_pred
        self.log.info("compare starts")

        pe_true=[]
        name = []
        pulse_num = []
        predict_pe_pulse_num = []
        predict_pe_pulse_percentage = []


        for x in pure_list:
            start = person_list.index(x)
            num = person_list.count(x)
            true_pe_state = true_label.loc[true_label["person_name"] == x]["PE_state"].mean()
            num, sums, precentage = get_detail_percentage(predict, start, num)

            name.append(x)
            pulse_num.append(sums)
            predict_pe_pulse_num.append(num)
            predict_pe_pulse_percentage.append(precentage)
            pe_true.append(true_pe_state)

        result = pd.DataFrame(data={"name": name, "pulse_num": pulse_num, "predict_pe_pulse_num": predict_pe_pulse_num,
                                    "predict_pe_pulse_percentage":predict_pe_pulse_percentage,"pe_true":pe_true})

        full_path = os.path.join(self.data_dir, "predict_result_by_person" ,Time_stamp().get_day())

        if not os.path.exists(full_path):
            os.makedirs(full_path)
        file_path = os.path.join(full_path, str(self.clf)+Time_stamp().get_time_stamp() + "_.csv")
        result.to_csv(file_path)
        self.log.info("predict_result_by_person saved to file "+file_path)

def get_detail_percentage(data, start, num):
    sum = 0
    for i in range(num):
        sum = sum + data[start + i]
    return num, sum, sum / num


if __name__=="__main__":
    from sklearn.svm import LinearSVC
    m=My_classifier(LinearSVC(C=1))
    m.dump()
