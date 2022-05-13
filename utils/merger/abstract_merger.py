#   -*- coding:utf-8 -*-
#   The abstract_merger.py in PE_CLF
#   created by Jiang Feng(silencejiang@zju.edu.cn)
#   created at 12:23 on 2022/5/13
import json
import os
import time
from utils.logger import Logger
from utils.project_dir import ProjectDir

class AbstractMerger:
    def __init__(self, title, prefix, suffix):
        self.data = []
        self.suffix = suffix
        root_dir = ProjectDir().dir_root
        self.target = os.path.join(root_dir, "input",prefix+time.strftime("%Y%m%d_%H%M%S", time.localtime())+".csv")
        self.is_header_wrote = False
        self.logger = Logger('clf').get_log()
        self.title = title
        self.feature_names = title.get_feature_names()
        self.field_names = title.get_title()

    def merge_dirs(self, target_dirs):
        for i in range(len(target_dirs)):
            target_dir = target_dirs[i]
            if os.path.isdir(target_dir):
                self.merge_dir(target_dir)
        self.logger.info('Finished processing all the dirs')

    def merge_dir(self, target_dir):
        files = os.listdir(target_dir)
        for i in range(len(files)):
            if files[i].endswith(self.suffix):
                path = os.path.join(target_dir,files[i])
                self.merge_file(path)
            else:
                self.logger.warning('Skip the non_json file %s in dir %s', files[i], target_dir)
        self.logger.info('Finished processing dir %s', target_dir)

    def merge_file(self, current_file):
        self.logger.warning("empty method! nothing is done to file %s",current_file)
        pass

    def read_file(self, file):
        if self.suffix == ".json":
            with open(file, "r", encoding="utf8") as np:
                self.data = json.load(np)
                self.logger.debug('Read json file %s finished.', file)
                np.close()