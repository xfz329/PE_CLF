#   -*- coding:utf-8 -*-
#   The merger.py in PE_CLF
#   created by Jiang Feng(silencejiang@zju.edu.cn)
#   created at 13:01 on 2022/5/13
import csv
from utils.merger.abstract_merger import AbstractMerger

class Merger(AbstractMerger):
    def __init__(self, title, prefix, suffix = ".json"):
        AbstractMerger.__init__(self, title, prefix, suffix)

    def merge_file(self, current_file):
        with open(self.target, "a", newline="") as csv_file:
            self.read_file(current_file)
            writer = csv.DictWriter(csv_file, fieldnames=self.field_names)

            if not self.is_header_wrote:
                writer.writeheader()
                self.is_header_wrote = True

            record = self.data['PPG Records']
            pulse = record['Pulses']

            out_list = [self.data['Version Num'], self.data['File Name'], self.data['Person Name'], self.data['PE State']]
            out_data = []
            for i in range(len(pulse)):
                out_data.clear()
                out_data.append(i + 1)
                features = pulse[i]['Features']
                # count = 0
                for j in range(len(features)):
                    current = features[j]
                    if  current['Abbr'] in self.feature_names:
                        # if current == "SVRR" & count==0 :
                        #     count = 1
                        # if current == "SVRR" & count==1 :
                        #     current = "CVRR"
                        out_data += current['Values']
                out_data = out_list + out_data
                out_dict = dict(zip(self.field_names, out_data))
                writer.writerow(out_dict)

if __name__ == '__main__':
    file = "D:\\UrgeData\\Documents\\Codes\\Graduate\\PE_CLF\\dataset\\data_version0.16.4\\Total\\NO\\by_wavePLETH_2017080113_2022_05_12_11_36_47.json"
    no = "D:\\UrgeData\\Documents\\Codes\\Graduate\\PE_CLF\\dataset\\data_version0.16.4\\Total\\NO"
    pe = "D:\\UrgeData\\Documents\\Codes\\Graduate\\PE_CLF\\dataset\\data_version0.16.4\\Total\\PE"

    dirs=[pe,no]
    from utils.titles.title import Title
    t = Title()
    x= Merger(t,prefix="mf")
    # x.read_json(file)
    # x.process_file(file)
    # x.process_dir(pe)
    x.merge_dirs(dirs)
