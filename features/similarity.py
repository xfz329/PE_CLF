#   -*- coding:utf-8 -*-
#   The similarity.py in PE_CLF
#   created by Jiang Feng(silencejiang@zju.edu.cn)
#   created at 6:41 on 2022/10/15

import pandas as pd
import numpy as np

def corr(a,b):
    x = np.array(a)
    y = np.array(b)
    mat = np.corrcoef(x,y)
    # print(mat)
    # print(mat[0][1])
    return mat[0][1]

def fd(a,b):
    ans = 0
    for i in range(len(a)):
        d = abs(a[i]-b[i])
        if d > ans:
            ans = d
    return ans

def area(a,b):
    ans = 0
    for i in range(len(a)):
        d = abs(a[i] - b[i])
        ans =  ans + d
    return ans