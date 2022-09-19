#   -*- coding:utf-8 -*-
#   The show_detail.py in PE_CLF
#   created by Jiang Feng(silencejiang@zju.edu.cn)
#   created at 20:55 on 2022/6/1
import os
import pandas as pd
import matplotlib.pyplot as plt
from data_loader.point_loader import PointLoader
from utils.project_dir import ProjectDir
from utils.logger import Logger


Color_0 = "#6699CC"
Color_1 = "#FF9900"
Color_2 = "#333333"

class Show_details:
    def __init__(self,detail,center,mark):
        self.work_dir = os.path.join(ProjectDir().dir_output,"clustering")
        self.figure_dir = os.path.join(ProjectDir().dir_figures,"clustering")
        self.center_file = os.path.join(self.work_dir, center)
        self.mark_file = os.path.join(self.work_dir, mark)
        self.log = Logger().get_log()
        pl = PointLoader()
        pl.load_data(detail)
        self.data = pl.data
        self.centers = pd.read_csv(self.center_file)
        self.marks = pd.read_csv(self.mark_file)

    def draw_details(self):
        # self.log.info(self.data)
        dirs = ["No","PE"]
        center_label_0 = self.centers.iloc[:, 1].tolist()
        center_label_0 = list(filter(check_positive, center_label_0))

        center_label_1 = self.centers.iloc[:, 2].tolist()
        center_label_1 = list(filter(check_positive, center_label_1))
        for pe in [0,1]:
            self.log.info(pe)
            group = self.data.loc[self.data['PE_state'] == pe]
            current_sub_dir = dirs[pe]
            full_dir = os.path.join(self.figure_dir,current_sub_dir)
            if not os.path.exists(full_dir):
                os.makedirs(full_dir)
            names = list(set(group['person_name'].to_numpy()))
            self.log.info(names)

            for name in names:
                person = self.data.loc[self.data['person_name'] == name]
                length = person.shape[0]
                plt.clf()
                y_len = 0
                for i in range(length):
                    data_of_piece = person.iloc[i,5:].values.tolist()
                    y_label = list(filter(check_positive, data_of_piece))
                    y_len = len(y_label)
                    x_label = range(len(y_label))
                    plt.plot(x_label, y_label, color=Color_0)

                plt.plot(range(len(center_label_0)),center_label_0,color=Color_1)
                plt.plot(range(len(center_label_1)),center_label_1,color=Color_2)

                fp_rate = self.marks.loc[self.marks["names"] == name,"fp_rate"].tolist()[0]
                pp_rate = self.marks.loc[self.marks["names"] == name,"pp_rate"].tolist()[0]

                plt.text(y_len*0.7,0.9,"fp_rate : {:.2%} \npp_rate : {:.2%}".format(1-fp_rate,pp_rate))
                title_name = name+" in group "+dirs[pe]
                plt.title(title_name)
                path = os.path.join(full_dir,title_name+".png")
                plt.savefig(path, format="png", dpi=300)



def check_positive(num):
    return num>0

if __name__ == "__main__":
    sd = Show_details(detail="0.17.0_mp_20220526_163806.csv",center="cluster_centers_20220601_153643_.csv",mark="cluster_stastitics_20220601_210749_.csv")
    sd.draw_details()