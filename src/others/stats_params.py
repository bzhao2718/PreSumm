from collections import defaultdict


class StatsParams(object):
    def __init__(self):
        self.dict_list = defaultdict(list)
        # self.accuracy = 0
        # self.elapse_time = 0.0
        # # self.sample_pred = None
        # self.cross_entropy = 0.0
        # self.ppl = 0.0

    def add_to_list(self, acc=0, elapse_time=0.0, cross_entropy=0.0, ppl=0.0):
        # if acc != 0:
        #     self.dict_list['accuracy'].append(acc)
        if elapse_time != 0.0:
            self.dict_list['elapsed_time'].append(elapse_time)
        # if cross_entropy != 0.0:
        #     self.dict_list['cross_entropy'].append(cross_entropy)
        # if ppl != 0.0:
        #     self.dict_list['ppl'].append(ppl)

    def get_elapse(self):
        return self.dict_list['elapsed_time']
