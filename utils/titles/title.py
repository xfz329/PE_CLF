
from utils.logger import Logger

class Title:
    def __init__(self):
        self.fixed = ["version", "file_name", "person_name", "PE_state", "Pulse"]
        self.feature_names = {
            "STDZ": 4,
            "LVA": 9,
            "LVS": 9,
            "LVLP": 9,
            "LVLR": 9,
            "LVLF": 9,
            "LVRR": 9,
            "LVRF": 9,
            "LVD": 9,
            "LVALR": 8,
            "LVALF": 8,
            "SVLR": 11,
            # "SVLF":22,
            "SVER": 11,
            # "SVEF":11,
            "SVAR": 10,
            "SVAF": 10,
            "SVAT": 10,
            "SVRR": 11,
            # "SVRF":11,
            "SVD": 11,
            "SVALR": 10,
            # "SVALF":10,
            "SVSR": 10,
            # "SVSF":10,
            "SVAAR": 10,
            # "SVAAF":10,
            "CVLR": 11,
            "CVLF": 11,
            "CVRR": 11,  # name SVRR
            "CVRF": 11,
            "CVD": 11,
            "CVALR": 10,
            "CVALF": 10,
            "CVAAR": 10,
            "CVAAF": 10
        }
        self.unmatched={
            "SVLF":22,
            "SVEF":11,
            "SVRF":11,
            "SVALF":10,
            "SVSF":10,
            "SVAAF":10,
        }
        self.title = self.fixed
        self.logger = Logger('clf').get_log()
        self.prepare_title()

    def get_feature_names(self):
        return self.feature_names

    def get_title(self):
        return self.title

    def get_unmatched(self):
        return self.unmatched

    def prepare_title(self):
        for i in self.feature_names:
            self.title += self.makeup(i)
        self.logger.debug(self.title)

    def makeup(self, key):
        n = self.feature_names.get(key)
        ans = []
        for i in range(n):
            ans.append(key + '_' + str(i + 1))
        return ans

if __name__=="__main__":
    t=Title()
    t.get_title()
    print(type(t.get_feature_names()))
    # print(t.get_feature_names()["SVAR"])