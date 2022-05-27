#   -*- coding:utf-8 -*-
#   The project_dir.py in PE_CLF
#   created by Jiang Feng(silencejiang@zju.edu.cn)
#   created at 15:50 on 2022/5/13
from pathlib import  Path
import os

class ProjectDir:
    def __init__(self):
        self.dir_root = Path(__file__).resolve().parent.parent
        self.dir_figures = self.join("figures")
        self.dir_input = self.join("input")
        self.dir_output = self.join("output")
        self.dir_logs = self.join("log")

    def join(self, subdir):
        full_dir = os.path.join(self.dir_root, subdir)
        if not os.path.exists(full_dir):
            os.makedirs(full_dir)
        return full_dir


if __name__ == "__main__":
    p =ProjectDir()