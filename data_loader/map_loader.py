#   -*- coding:utf-8 -*-
#   The map_loader.py in PE_CLF
#   created by Jiang Feng(silencejiang@zju.edu.cn)
#   created at 21:37 on 2022/5/14
import os
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from data_loader.abstract_data_loader import  AbstractDataLoader

class MapLoader(AbstractDataLoader):

    def __init__(self):
        AbstractDataLoader.__init__(self)
        self.X = None
        self.y = None


    def load_data(self, file , file_path = None):
        if file_path is None:
            file_path = self.input_dir
        full_path = os.path.join(file_path,file)
        self.data = pd.read_csv(full_path)
        self.construct()
        return self.split()

    def construct(self):
        x = self.data.shape[0]
        self.X = np.zeros((x, 140, 120))
        self.y = np.zeros(x)

        for i in range(x):
            self.y[i] = self.data.iloc[i,3]
            for j in range(140 - 20):
                point = self.data.iloc[i,j +5]
                if point != -1:
                    k = int(100*point)
                    self.X[i][j+10][k+10] = 1.0

    def split(self):
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size = 0.20, random_state = 42)
        return X_train, X_test, y_train, y_test

    def show_ppg(self, idx):
        data = 255 * np.rot90(self.X[idx],k=1)
        plt.imshow(data, cmap= matplotlib.cm.binary, interpolation= "nearest")
        plt.title("index is "+str(idx) +"; label is "+str(self.y[idx]))
        plt.axis("off")
        plt.show()
        self.log.info(self.y[idx])


if __name__=="__main__":
    ml = MapLoader()
    ml.load_data("0.17.0_mp_20220526_163806.csv")
    for i in range(198,204):
        ml.show_ppg(i)