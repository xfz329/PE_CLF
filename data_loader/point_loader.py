#   -*- coding:utf-8 -*-
#   The point_loader.py in PE_CLF
#   created by Jiang Feng(silencejiang@zju.edu.cn)
#   created at 14:42 on 2022/5/16
import os
import pandas as pd

from data_loader.abstract_data_loader import  AbstractDataLoader

class PointLoader(AbstractDataLoader):

    def __init__(self):
        AbstractDataLoader.__init__(self)
        self.X = None
        self.y = None


    def load_data(self, file = "mp20220513_194847.csv", file_path = None):
        if file_path is None:
            file_path = self.input_dir
        full_path = os.path.join(file_path,file)
        self.data = pd.read_csv(full_path)
        self.construct()
        return self.split()

    def construct(self):
        self.X = self.data.iloc[:,5:]
        self.y = self.data.iloc[:,3]

    def split(self):
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size = 0.20, random_state = 42)
        return X_train, X_test, y_train, y_test


if __name__=="__main__":
    pl = PointLoader()
    pl.load_data()