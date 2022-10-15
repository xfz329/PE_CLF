#   -*- coding:utf-8 -*-
#   The merge_detect_2.py in PE_CLF
#   created by Jiang Feng(silencejiang@zju.edu.cn)
#   created at 6:36 on 2022/9/30
import csv

import numpy as np
import scipy.interpolate as spi
import matplotlib.pyplot as plt

from utils.merger.abstract_merger import AbstractMerger


class MergeDetect2(AbstractMerger):
    def __init__(self, title, prefix, suffix = ".json", normalization_range = (0,1),normalization_length = 100):
        AbstractMerger.__init__(self, title, prefix, suffix)
        self.normalization_range = normalization_range
        self.normalization_length = normalization_length
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

                if length == 100 :
                    self.logger.info("the sample point of  %dth pulse in file %s is  100, don't need re-sampling", i, current_file)

                target_num = 100 * length

                x = np.linspace(0,length-1,length)
                xx = np.linspace(0,100-1,target_num)
                tck = spi.splrep(x, points, k=3)
                yy = spi.splev(xx,tck)
                y_list = yy.tolist()

                end_value = points[-1]
                # print(end_value)

                abs_y = [abs(k- end_value) for k in y_list]
                sub_y = abs_y[-target_num//2:-1]

                new_length = target_num//2 +sub_y.index(min(sub_y))

                new_point = []
                for j in range(0,new_length,new_length//100):
                    new_point.append(yy[j])
                # new_point = new_point[0:100]

                # plt.figure()
                # plt.plot(range(len(points)), points)
                # plt.title(str(i))
                # plt.show()
                #
                # plt.figure()
                # plt.plot(xx, yy)
                # plt.title(str(i))
                # plt.show()
                #
                # plt.figure()
                # plt.plot(range(len(new_point)), new_point)
                # plt.title(str(i))
                # plt.show()

                points_normal = self.normalization(new_point)
                for j in range(100):
                    out_data.append(points_normal[j][0])
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




if __name__ == '__main__':
    import os
    from utils.data_version import Version

    root = "D:\\UrgeData\\Documents\\Codes\\Graduate\\PE_PPG"
    version_short = Version().current_data_version()
    version = "data_version" + version_short
    path_type = "Detection"
    no = os.path.join(root, version, path_type, "NO")
    pe = os.path.join(root, version, path_type, "PE")

    dirs = [no, pe]

    from utils.titles.title_detect import TitleDetect
    t = TitleDetect(100)
    x= MergeDetect2(t, prefix=version_short+"_mp2_")
    x.merge_dirs(dirs)