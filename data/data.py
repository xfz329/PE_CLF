#   -*- coding:utf-8 -*-
#   The data.py in my_hands_ml
#   created by Jiang Feng(silencejiang@zju.edu.cn)
#   created at 18:56 on 2022/4/15
import os
import pandas as pd
from utils.logger import Logger

class Data:
    root_dir = "."
    dir = "input"
    csv_path = os.path.join(root_dir, dir)

    def __init__(self):
        self.log = Logger("clf").get_log()
        self.data = None
        os.makedirs(self.csv_path, exist_ok=True)

    def load_data(self, fname, fpath=csv_path):
        csv_full_path = os.path.join(fpath, fname)
        self.data = pd.read_csv(csv_full_path)
        return self.data

    def is_column_duplicated(self,cl):
        idx = self.data.index
        s = set()
        for i in idx:
            s.add(self.data[cl].get(i))
            if len(s) > 1:
                return False
        return True

    def get_duplicated_columns(self):
        cl = self.data.columns
        res = []
        for c in cl:
            if self.is_column_duplicated(c):
                res.append(c)
        return res

    def drop_invalid_value(self):
        cl = self.get_duplicated_columns()
        cl.append("file_name")
        cl.append("person_name")
        cl.append("Pulse")
        self.log.info(cl)
        self.data.info()
        self.data.drop(columns = cl,inplace = True)
        self.data.info()
        return self.data