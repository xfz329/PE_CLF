#   -*- coding:utf-8 -*-
#   The feature_loader.py in PE_CLF
#   created by Jiang Feng(silencejiang@zju.edu.cn)
#   created at 20:11 on 2022/5/14
import os
import pandas as pd
import numpy as np
from data_loader.abstract_data_loader import AbstractDataLoader
from utils.time_stamp import Time_stamp

class FeatureLoader(AbstractDataLoader):
    def __init__(self):
        AbstractDataLoader.__init__(self)
        self.data_all = None

    def load_data(self, file, file_path = None):
        if file_path is None:
            file_path = self.input_dir
        full_path = os.path.join(file_path,file)
        self.data_all = pd.read_csv(full_path)
        self.drop_invalid_value()
        return self.data

    def drop_invalid_value(self):
        cl = self.get_duplicated_columns()
        cl.append("file_name")
        cl.append("person_name")
        cl.append("Pulse")
        self.log.info(cl)
        self.data_all.info()
        self.data=self.data_all.drop(columns=cl, inplace=False)
        self.data.info()

    def is_column_duplicated(self,cl):
        idx = self.data_all.index
        s = set()
        for i in idx:
            s.add(self.data_all[cl].get(i))
            if len(s) > 1:
                return False
        return True

    def get_duplicated_columns(self):
        cl = self.data_all.columns
        res = []
        for c in cl:
            if self.is_column_duplicated(c):
                res.append(c)
        return res

    def split(self,stick = True):
        full_path = os.path.join(self.input_dir, "pulse_split.npz")
        data = self.data
        if not stick:
            from sklearn.model_selection import StratifiedShuffleSplit

            split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
            for train_index, test_index in split.split(data, data["PE_state"]):
                train_set = data.loc[train_index]
                test_set = data.loc[test_index]
            np.savez(full_path,train_index=train_index,test_index=test_index)

        else:
            npzfile = np.load(full_path)
            train_index = npzfile['train_index']
            test_index = npzfile['test_index']
            train_set = data.loc[train_index]
            test_set = data.loc[test_index]

        self.log.info(train_index)
        self.log.info(test_index)

        X_train = train_set.drop("PE_state", axis=1)
        y_train = train_set["PE_state"].copy()
        X_test = test_set.drop("PE_state", axis=1)
        y_test = test_set["PE_state"].copy()

        # from sklearn.preprocessing import StandardScaler
        # std_scaler = StandardScaler()
        # X_train = std_scaler.fit_transform(X_train)
        # X_test = std_scaler.fit_transform(X_test)
        return X_train,X_test,y_train,y_test

    def split_by_person(self, stick = True):
        full_path = os.path.join(self.input_dir, "person_split.npz")
        if not stick:
            from sklearn.model_selection import train_test_split
            train =[]
            test = []

            health = self.data_all.loc[self.data_all["PE_state"] == 0]
            health_person_list = health["person_name"].values.tolist()
            health_person_list = list(set(health_person_list))
            health_person_name = pd.DataFrame(health_person_list,columns=["person_name"])

            pe = self.data_all.loc[self.data_all["PE_state"] == 1]
            pe_person_list = pe["person_name"].values.tolist()
            pe_person_list = list(set(pe_person_list))
            pe_person_name = pd.DataFrame(pe_person_list, columns=["person_name"])

            train_name,test_name = train_test_split(health_person_name,test_size=0.2,random_state=42)
            for x in train_name.values.tolist():
                train = train + x
            for x in test_name.values.tolist():
                test = test + x

            train_name, test_name = train_test_split(pe_person_name, test_size=0.2, random_state=42)
            for x in train_name.values.tolist():
                train = train + x
            for x in test_name.values.tolist():
                test = test + x
            np.savez(full_path, train=train, test=test)

        else:
            npzfile = np.load(full_path)
            train = npzfile['train']
            test = npzfile['test']
        self.log.info(train)
        self.log.info(test)
        train_set=self.data_all.loc[self.data_all["person_name"].isin(train)]
        test_set = self.data_all.loc[self.data_all["person_name"].isin(test)]

        X_train = train_set.drop(["PE_state","file_name","person_name","Pulse","version"], axis=1,inplace=False)
        y_train = train_set["PE_state"].copy()
        X_test = test_set.drop(["PE_state","file_name","person_name","Pulse","version"], axis=1,inplace=False)
        y_test = test_set["PE_state"].copy()

        return X_train, X_test, y_train, y_test,train_set,test_set


    def corr(self):
        corr_matrix = self.data.corr()
        pe_corr = corr_matrix["PE_state"]#.sort_values(ascending=False)
        self.log.info(pe_corr)
        full_path = os.path.join(self.output_dir, "pe_corr" + Time_stamp().get_time_stamp() + "_.csv")
        pe_corr.to_csv(full_path)

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

        full_path = os.path.join(self.output_dir, "stastitics_"+Time_stamp().get_time_stamp()+"_.csv")
        result.to_csv(full_path)
        self.log.info("stastitics finished!")

if __name__ == "__main__":
    fd = FeatureLoader()
    data = fd.load_data("0.17.0_mf_20220526_163827.csv")
    fd.split_by_person()
    fd.split()
    # fd.non_para_analyze()
    # fd.corr()