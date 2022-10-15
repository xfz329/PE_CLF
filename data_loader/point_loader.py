#   -*- coding:utf-8 -*-
#   The point_loader.py in PE_CLF
#   created by Jiang Feng(silencejiang@zju.edu.cn)
#   created at 14:42 on 2022/5/16
import os
import pandas as pd
import numpy as np

from data_loader.abstract_data_loader import  AbstractDataLoader

class PointLoader(AbstractDataLoader):

    def __init__(self):
        AbstractDataLoader.__init__(self)
        self.data_all = None

    def load_data(self, file, file_path=None):
        if file_path is None:
            file_path = self.input_dir
        full_path = os.path.join(file_path, file)
        self.data_all = pd.read_csv(full_path)
        self.drop_invalid_value()
        return self.data

    def drop_invalid_value(self):
        cl = []
        cl.append("version")
        cl.append("file_name")
        cl.append("person_name")
        cl.append("Pulse")
        self.log.info(cl)
        self.data_all.info()
        self.data = self.data_all.drop(columns=cl, inplace=False)
        self.data = self.data.replace(np.nan, 0)
        self.data.info()

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

        X_train = train_set.drop(["PE_state", "file_name", "person_name", "Pulse", "version"], axis=1, inplace=False)
        y_train = train_set["PE_state"].copy()
        X_test = test_set.drop(["PE_state", "file_name", "person_name", "Pulse", "version"], axis=1, inplace=False)
        y_test = test_set["PE_state"].copy()

        return X_train, X_test, y_train, y_test, train_set, test_set


if __name__=="__main__":
    pl = PointLoader()
    pl.load_data("0.17.0_mp_20221013_032707.csv")