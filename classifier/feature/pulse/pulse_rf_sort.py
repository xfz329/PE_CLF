#   -*- coding:utf-8 -*-
#   The pulse_rf_sort.py in PE_CLF
#   created by Jiang Feng(silencejiang@zju.edu.cn)
#   created at 17:12 on 2022/10/8
import pandas as pd

class RandomForestPercentage:
    def __init__(self,full_path = r"D:\UrgeData\Documents\Codes\Graduate\PE_CLF\output\rf\20221008\RandomForestClassifier(random_state=42)20221008_172021_.csv"):
        self.data = pd.read_csv(full_path)
        self.sort = self.data.sort_values(["importance","feature"],ascending=[False,True],inplace=False)

    def get(self,percentage=100):
        p_sum = 0
        f_sum = []
        for index, row in self.sort.iterrows():
            if percentage < 100:
                if p_sum > percentage:
                    break
            p_sum = p_sum + row['importance']*100
            f_sum.append(row['feature'])

        return f_sum



if __name__=="__main__":
    rfp = RandomForestPercentage()
    print(len(rfp.get()))
    print(len(rfp.get(70)))
    print(len(rfp.get(50)))
    print(len(rfp.get(30)))
