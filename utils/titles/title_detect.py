#   -*- coding:utf-8 -*-
#   The title_detect.py in PE_CLF
#   created by Jiang Feng(silencejiang@zju.edu.cn)
#   created at 16:38 on 2022/5/13

from utils.logger import Logger

class TitleDetect:
    def __init__(self):
        self.fixed = ["version", "file_name", "person_name", "PE_state", "Pulse"]
        self.feature_names = {
            "P": 120,
        }
        self.title = self.fixed
        self.logger = Logger('clf').get_log()
        self.prepare_title()

    def get_feature_names(self):
        return self.feature_names

    def get_title(self):
        return self.title

    def prepare_title(self):
        for i in self.feature_names:
            self.title += self.makeup(i)
        self.logger.debug(self.title)

    def makeup(self, key):
        n = self.feature_names.get(key)
        ans = []
        for i in range(n):
            ans.append(key + '_' + str(i + 1).zfill(3))
        return ans


if __name__ == "__main__":
    t = TitleDetect()
    # print(t.get_title())
    print(type(t.get_feature_names()))
    # print(t.get_feature_names()["SVAR"])