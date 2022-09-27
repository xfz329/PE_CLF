#   -*- coding:utf-8 -*-
#   The abstract_data_loader.py in PE_CLF
#   created by Jiang Feng(silencejiang@zju.edu.cn)
#   created at 20:06 on 2022/5/14
from abc import ABCMeta, abstractmethod
from utils.logger import Logger
from utils.project_dir import ProjectDir

class AbstractDataLoader(object, metaclass= ABCMeta):
    def __init__(self):
        self.log = Logger("clf").get_log()
        self.input_dir = ProjectDir().dir_input
        self.output_dir = ProjectDir().dir_output
        self.data = None

    @abstractmethod
    def load_data(self, file):
        pass

    @abstractmethod
    def split(self):
        pass

    @abstractmethod
    def split_by_person(self):
        pass