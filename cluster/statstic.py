#   -*- coding:utf-8 -*-
#   The statstic.py in PE_CLF
#   created by Jiang Feng(silencejiang@zju.edu.cn)
#   created at 20:01 on 2022/5/30
import os
import pandas as pd
from data_loader.cluster_loader import Cluster_loader
from utils.logger import Logger
from utils.project_dir import ProjectDir
from utils.time_stamp import Time_stamp
class Cluster_stastics:
    def __init__(self):
        self.data = None
        self.log = Logger().get_log()
        pass

    def analyze(self):
        cl = Cluster_loader()
        self.data = cl.load_data("0.17.0_cluster_detail.csv",r"D:\UrgeData\Documents\Codes\Graduate\PE_CLF\output\clustering")
        pe = self.data.loc[self.data['PE_state']==1]
        no = self.data.loc[self.data['PE_state']==0]
        pe_names = list(set(pe['person_name'].values))
        no_names = list(set(no['person_name'].values))
        # self.log.info(pe_names)
        # self.log.info(no_names)
        names = []
        pe_state = []
        fp_0 = []
        fp_1 = []
        fp_rate = []
        pp_0 = []
        pp_1 = []
        pp_rate = []

        current_pe = -1
        for group in [no_names, pe_names]:
            current_pe += 1
            for name in group:
                person = self.data.loc[self.data['person_name']==name]
                shape = person.shape
                pulse_num = shape[0]
                # self.log.info(person)
                # self.log.info(shape)
                current_fp_0 = person.iloc[:,4:5].sum(axis=0)[0].astype(int)
                current_fp_1 = pulse_num - current_fp_0
                current_fp_rate = current_fp_1/pulse_num

                current_pp_0 = person.iloc[:, 5:6].sum(axis=0)[0].astype(int)
                current_pp_1 = pulse_num - current_pp_0
                current_pp_rate = current_pp_1 / pulse_num

                names.append(name)
                pe_state.append(current_pe)
                fp_0.append(current_fp_0)
                fp_1.append(current_fp_1)
                fp_rate.append(current_fp_rate)
                pp_0.append(current_pp_0)
                pp_1.append(current_pp_1)
                pp_rate.append(current_pp_rate)

        result = pd.DataFrame(data={"names": names, "pe_state": pe_state, "fp_0": fp_0,"fp_1":fp_1,"fp_rate":fp_rate,"pp_0": pp_0,"pp_1":pp_1,"pp_rate":pp_rate})
        full_path = os.path.join(ProjectDir().dir_output, "clustering","cluster_stastitics_" + Time_stamp().get_time_stamp() + "_.csv")
        result.to_csv(full_path)









if __name__ == "__main__":
    cs = Cluster_stastics()
    cs.analyze()
