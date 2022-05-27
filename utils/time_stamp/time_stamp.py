#   -*- coding:utf-8 -*-
#   The time_stamp.py in PE_CLF
#   created by Jiang Feng(silencejiang@zju.edu.cn)
#   created at 13:41 on 2022/5/27
import time
class Time_stamp:
    def __init__(self):
        pass

    def get_time_stamp(self):
        return time.strftime("%Y%m%d_%H%M%S", time.localtime())