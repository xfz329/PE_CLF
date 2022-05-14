#   -*- coding:utf-8 -*-
#   The test.py in PE_CLF
#   created by Jiang Feng(silencejiang@zju.edu.cn)
#   created at 13:56 on 2022/4/27
from data.data import  Data

d= Data()
res=d.load_data("mf20220513_193751.csv")
print(res.info())

from sklearn.feature_selection import VarianceThreshold

sel = VarianceThreshold()
res.drop(columns = ["file_name","person_name"],inplace = True)
res = sel.fit_transform(res)
print(res)
