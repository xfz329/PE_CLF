#   -*- coding:utf-8 -*-
#   The calculator.py in PE_CLF
#   created by Jiang Feng(silencejiang@zju.edu.cn)
#   created at 6:49 on 2022/10/15

import os
import pandas as pd
import numpy as np

from utils.logger import Logger
from utils.project_dir import ProjectDir
from utils.merger.merge_detect_2 import MergeDetect2
from utils.time_stamp import Time_stamp

from features.similarity import *

class Calculator:
    def __init__(self):
        self.log = Logger("clf").get_log()
        self.input_dir = ProjectDir().dir_input
        self.output_dir = ProjectDir().dir_output
        self.current_dir = None
        self.data = None

    def load_data(self, kind, file, file_path=None):
        if file_path is None:
            file_path = self.input_dir
        full_path = os.path.join(file_path, file)
        self.data = pd.read_csv(full_path)
        self.current_dir = os.path.join(self.output_dir, "3rd_feature", Time_stamp().get_day(),kind+Time_stamp().get_time_stamp())
        if not os.path.exists(self.current_dir):
            os.makedirs(self.current_dir)

        self.select_pe_state(0)
        self.select_pe_state(1)

    def select_pe_state(self, state):
        people_with_state = self.data.loc[self.data["PE_state"] == state]
        name_list = people_with_state["person_name"].values.tolist()
        name_list = list(set(name_list))
        for name in name_list:
            self.select_person(state,name)


    def select_person(self, state, name):
        person = self.data.loc[self.data["person_name"] == name]
        position_list =  person.index.tolist()
        c, f, a = self.calculate_each_person(position_list)
        result = pd.DataFrame({'corr': c, 'fd': f, 'area':a})
        file_path = os.path.join(self.current_dir,  str(state) +"_" + name + "_.csv")
        result.to_csv(file_path)


    def calculate_each_person(self, p_list):
        list_len = len(p_list)
        ave = self.get_average_pulse(p_list[0])
        co_total = []
        fd_total = []
        area_total = []
        for i in range(list_len):
            current = self.get_specific_pulse_in_line(p_list[i])
            co_total.append(corr(ave,current))
            fd_total.append(fd(ave,current))
            area_total.append(area(ave,current))
        return co_total,fd_total,area_total

    def get_average_pulse(self, start, num = 10):
        temp = self.get_specific_pulse_in_line(start)
        pulse_len = len(temp)
        for i in range(1,num):
            current = self.get_specific_pulse_in_line(start+i)
            for j in range(pulse_len):
                temp[j] = temp[j]+current[j]
        result = normalization(temp)
        return result

    def get_specific_pulse_in_line(self,line):
        pulse = self.data.iloc[line,5:].values.tolist()
        # print(pulse)
        return pulse

def normalization(x, n_range=(0,1)):
    from sklearn.preprocessing import MinMaxScaler
    X = np.array(x).reshape(-1,1)
    ans = MinMaxScaler(feature_range= n_range).fit_transform(X).tolist()
    result = []
    for i in range(len(ans)):
        result.append(ans[i][0])
    return result

if __name__=="__main__":
    c = Calculator()
    c.load_data("re_sample","0.17.0_mp2_20221013_004934.csv")
    c.load_data("add","0.17.0_mp_20221013_032707.csv")
