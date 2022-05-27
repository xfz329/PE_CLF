#   -*- coding:utf-8 -*-
#   The my_cluster.py in PE_CLF
#   created by Jiang Feng(silencejiang@zju.edu.cn)
#   created at 20:03 on 2022/5/26
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

import pandas as pd
from sklearn.cluster import KMeans

from data_loader.feature_loader import FeatureLoader
from data_loader.point_loader import PointLoader
from utils.data_version import Version
from utils.logger import Logger
from utils.project_dir import ProjectDir
from utils.time_stamp import Time_stamp

Color_0 = "#6699CC"
Color_1 = "#99CCFF"


class My_cluster:
    def __init__(self):
        self.out_dir = os.path.join(ProjectDir().dir_output, "clustering")
        self.ans = None
        self.X_f = None
        self.X_p = None
        self.y_f = None
        self.y_p = None
        self.fp = None
        self.pp = None
        self.log = Logger().get_log()

    def load_data(self):
        df = FeatureLoader().load_data("0.17.0_mf_20220526_163827.csv")
        self.X_f = df.drop("PE_state", axis=1)
        self.y_f = df["PE_state"].copy()
        self.X_p , self.y_p = PointLoader().load_data("0.17.0_mp_20220526_163806.csv")

    def fit_predict(self):
        self.load_data()
        self.log.info("judge the feature")
        kmeans = KMeans(n_clusters=2)
        self.fp = kmeans.fit_predict(self.X_f)
        labels = kmeans.labels_
        self.judge_outer(self.y_f, self.fp)
        self.judge_inner(self.X_f,labels)
        self.draw("features",self.X_p,self.fp)


        self.log.info("judge the points")
        self.pp = kmeans.fit_predict(self.X_p)
        labels = kmeans.labels_
        self.judge_outer(self.y_p, self.pp)
        self.judge_inner(self.X_p, labels)
        self.draw("points", self.X_p, self.pp)


    def draw(self, title, data, pred):
        # fig = plt.figure()
        # plt.title(title)
        plt.clf()
        ax = plt.axes(projection='3d')
        length = len(data)

        front = -1
        back = -1
        for i in range(len(pred)):
            data_of_piece = data.loc[i].values.tolist()
            z = list(filter(check_minus_1,data_of_piece))
            x = range(len(z))
            if pred[i] == 1:
                front += 1
                y = [front]*len(z)
                color_type = Color_1
            else:
                back += 1
                y = [length-1-back]*len(z)
                color_type = Color_0
            ax.plot(x,y,z,color=color_type)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_title('3D contour')
        plt.show()






    def draw_single(self):
        plt.title("1")
        for i in range(len(self.y_p)):
            data = self.X_p.loc[i].values.tolist()
            y_label = list(filter(check_minus_1,data))
            # print(y_label)
            x_label = range(len(y_label))
            color_type = Color_0
            if self.pp[i] == 1:
                color_type = Color_1
            plt.plot(x_label,y_label,color = color_type)
        plt.show()

    def judge_outer(self,true_label,pred_label):
        from sklearn.metrics import confusion_matrix,adjusted_rand_score
        from sklearn.metrics import adjusted_mutual_info_score,homogeneity_score
        from sklearn.metrics import completeness_score,v_measure_score,fowlkes_mallows_score

        cm = confusion_matrix(true_label, pred_label)
        self.log.info("cm")
        self.log.info(cm)

        ari = adjusted_rand_score(true_label, pred_label)
        self.log.info("ARI")
        self.log.info(ari)

        ami = adjusted_mutual_info_score(true_label, pred_label)
        self.log.info("AMI")
        self.log.info(ami)

        hs = homogeneity_score(true_label, pred_label)
        self.log.info("homogeneity")
        self.log.info(hs)

        cs = completeness_score(true_label, pred_label)
        self.log.info("completeness")
        self.log.info(cs)

        vm = v_measure_score(true_label, pred_label)
        self.log.info("V_measure")
        self.log.info(vm)

        fmi = fowlkes_mallows_score(true_label, pred_label)
        self.log.info("fmi")
        self.log.info(fmi)

    def judge_inner(self,X,labels):
        from sklearn.metrics import silhouette_score

        ss = silhouette_score(X,labels, metric="cosine")
        self.log.info("ss+cosine")
        self.log.info(ss)

        ss = silhouette_score(X, labels, metric="euclidean")
        self.log.info("ss+euclidean")
        self.log.info(ss)

    def save(self):
        diff = self.y_f - self.y_p
        self.ans = pd.DataFrame(data={"y_f": self.y_f, "y_p": self.y_p, "diff": diff, "fp": self.fp, "pp": self.pp})
        file_path = os.path.join(self.out_dir, Version().current_data_version()+"_cluster_"+Time_stamp().get_time_stamp()+".csv")
        self.ans.to_csv(file_path)




def check_minus_1(x):
    return x != -1




if __name__ == "__main__":
    mc =My_cluster()
    mc.fit_predict()
    mc.draw_single()
    # mc.save()


