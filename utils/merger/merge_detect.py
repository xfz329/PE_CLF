#   -*- coding:utf-8 -*-
#   The merge_detect.py in PE_CLF
#   created by Jiang Feng(silencejiang@zju.edu.cn)
#   created at 16:28 on 2022/5/13
import csv
import numpy as np
from utils.merger.abstract_merger import AbstractMerger

class MergeDetect(AbstractMerger):
    def __init__(self, title, prefix, suffix = ".json", normalization_range = (0,1)):
        AbstractMerger.__init__(self, title, prefix, suffix)
        self.normalization_range = normalization_range
        self.fill_num = 0

    def merge_file(self, current_file):
        with open(self.target, "a", newline="") as csv_file:
            self.read_file(current_file)
            writer = csv.DictWriter(csv_file, fieldnames=self.field_names)

            if not self.is_header_wrote:
                writer.writeheader()
                self.is_header_wrote = True

            record = self.data['PPG Records original']
            pulse = record['Pulses']

            out_list = [self.data['Version Num'], self.data['File Name'], self.data['Person Name'], self.data['PE State']]
            out_data = []
            for i in range(len(pulse)):
                out_data.clear()
                out_data.append(i + 1)
                points = pulse[i]['pointsBasic']
                length = len(points)
                if length > 120:
                    self.logger.error("max pulse points exceeds 120 for %dth pulse in file %s",i,current_file)
                    return
                points_normal = self.normalization(points)
                for j in range(length):
                    out_data.append(points_normal[j][0])
                for j in range(120-length):
                    out_data.append(self.fill_num)
                out_data = out_list + out_data
                out_dict = dict(zip(self.field_names, out_data))
                writer.writerow(out_dict)

    def normalization(self, x):
        from sklearn.preprocessing import MinMaxScaler
        X = np.array(x).reshape(-1,1)
        ans = MinMaxScaler(feature_range= self.normalization_range).fit_transform(X)
        # self.logger.info(ans.shape)
        # self.logger.info(ans.tolist())
        return ans.tolist()

    def set_fill_num(self,num):
        self.fill_num = num



if __name__ == '__main__':
    dt = "D:\\UrgeData\\Documents\\Codes\\Graduate\\PE_CLF\\dataset\\data_version0.16.4\\Detection"

    from utils.titles.title_detect import TitleDetect
    t = TitleDetect()
    x= MergeDetect(t, prefix="mp")
    x.set_fill_num(-1)
    x.merge_dir(dt)