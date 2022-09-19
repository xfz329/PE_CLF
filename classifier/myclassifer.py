#   -*- coding:utf-8 -*-
#   The myclassifer.py in PE_CLF
#   created by Jiang Feng(silencejiang@zju.edu.cn)
#   created at 21:11 on 2022/4/24
import  matplotlib.pyplot as plt
from utils.logger import Logger
from utils.project_dir import ProjectDir
import os

class My_classifier:
    def __init__(self, clf,**kwargs):
        self.out_dir = ProjectDir().dir_figures
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

    def set_Datasets(self,X_train,y_train,X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def fit(self):
        self.clf.fit(self.X_train,self.y_train)

    def predict(self):
        self.y_test_pred = self.clf.predict(self.X_test)

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

    def fit_predict(self):
        import time

        start = time.time()
        self.fit()
        self.performance_judge()
        self.predict()
        self.performance_judge(on_test_set=True)
        end = time.time()
        self.log.info("time consuming " + str(end - start))



