#   -*- coding:utf-8 -*-
#   The feature_loader.py in PE_CLF
#   created by Jiang Feng(silencejiang@zju.edu.cn)
#   created at 20:11 on 2022/5/14
import os
import pandas as pd
from data_loader.abstract_data_loader import AbstractDataLoader

class FeatureLoader(AbstractDataLoader):
    def __init__(self):
        AbstractDataLoader.__init__(self)

    def load_data(self, file, file_path = None):
        if file_path is None:
            file_path = self.input_dir
        full_path = os.path.join(file_path,file)
        self.data = pd.read_csv(full_path)
        self.drop_invalid_value()
        return self.data

    def drop_invalid_value(self):
        cl = self.get_duplicated_columns()
        cl.append("file_name")
        cl.append("person_name")
        cl.append("Pulse")
        self.log.info(cl)
        self.data.info()
        self.data.drop(columns=cl, inplace=True)
        self.data.info()

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

    def non_para_analyze(self):
        import pingouin as pg

        dvs = self.data.drop("PE_state", axis = 1).columns
        health = self.data.loc[self.data["PE_state"] == 0]
        pe = self.data.loc[self.data["PE_state"] == 1]

        varibles = []
        mwu = []
        kruskal = []
        for dv in dvs:
            varibles.append(dv)
            ans = pg.kruskal(self.data , dv = dv, between = "PE_state" , detailed= True)
            kruskal.append(ans.loc["Kruskal","p-unc"].round(3))
            ans = pg.mwu(health[dv],pe[dv])
            mwu.append(ans.loc["MWU","p-val"].round(3))

        result = pd.DataFrame(data = {"varibles":varibles , "mwu":mwu,"kruskal":kruskal})

        full_path = os.path.join(self.output_dir, "stastitics_"+self.get_time_stamp()+"_.csv")
        result.to_csv(full_path)
        self.log.info("stastitics finished!")

if __name__ == "__main__":
    fd = FeatureLoader()
    data = fd.load_data("mf20220513_193751.csv")
    fd.non_para_analyze()