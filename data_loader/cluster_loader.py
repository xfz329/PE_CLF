#   -*- coding:utf-8 -*-
#   The cluster_loader.py in PE_CLF
#   created by Jiang Feng(silencejiang@zju.edu.cn)
#   created at 20:05 on 2022/5/30
import pandas as pd
import os
from data_loader.abstract_data_loader import AbstractDataLoader
class Cluster_loader(AbstractDataLoader):
    def __init__(self):
        AbstractDataLoader.__init__(self)

    def load_data(self, file, file_path=None):
        if file_path is None:
            file_path = self.input_dir
        full_path = os.path.join(file_path, file)
        self.data = pd.read_csv(full_path)
        return self.data

    def split(self):
        pass