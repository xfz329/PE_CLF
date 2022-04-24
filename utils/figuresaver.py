#   -*- coding:utf-8 -*-
#   The figuresaver.py in PE_CLF
#   created by Jiang Feng(silencejiang@zju.edu.cn)
#   created at 16:40 on 2022/4/22
import os
import matplotlib.pyplot as plt
from utils.logger import Logger


class FigureSaver:
    root_dir = "."
    dir = "figures"
    image_path = os.path.join(root_dir, dir)

    def __init__(self):
        self.log = Logger("clf").get_log()
        os.makedirs(self.image_path, exist_ok=True)

    def save_fig(self, fig_id, tight_layout=True, fig_extension="png", resolution=300):
        path = os.path.join(self.image_path, fig_id + "." + fig_extension)
        self.log.info("Saving figure", fig_id)
        if tight_layout:
            plt.tight_layout()
        plt.savefig(path, format=fig_extension, dpi=resolution)

    def plot_roc_curve(self,fpr, tpr, filename, label=None):
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, linewidth=2, label=label)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.axis([0, 1, 0, 1])
        plt.xlabel('False Positive Rate', fontsize=16)
        plt.ylabel('True Positive Rate', fontsize=16)
        # self.save_fig(filename)
        plt.show()