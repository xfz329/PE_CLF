#   -*- coding:utf-8 -*-
#   The data.py in my_hands_ml
#   created by Jiang Feng(silencejiang@zju.edu.cn)
#   created at 18:56 on 2022/4/15
import os
import pandas as pd
import pingouin as pg
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

    def non_para_analyze(self):
        between = self.data["PE_state"].copy()
        dvs = self.data.drop("PE_state", axis = 1)
        health = self.data.loc[self.data["PE_state"] == 0]
        pe = self.data.loc[self.data["PE_state"] == 1]
        varibles = []
        mwu = []
        kruskal = []
        for dv in dvs.columns:
            varibles.append(dv)
            ans = pg.kruskal(self.data , dv = dv, between = "PE_state" , detailed= True)
            kruskal.append(ans.loc["Kruskal","p-unc"].round(3))
            ans = pg.mwu(health[dv],pe[dv])
            mwu.append(ans.loc["MWU","p-val"].round(3))

        result = pd.DataFrame(data = {"varibles":varibles , "mwu":mwu,"kruskal":kruskal})
        csv_full_path = os.path.join(Data.csv_path, "stastitics.csv")
        result.to_csv(csv_full_path)
        self.log.info("stastitics finished!")