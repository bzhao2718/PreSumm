from collections import defaultdict
import neptune


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


class NeptuneLogger(object):
    def __init__(self):
        self.prj_TestPrj = 'bzhao271828/TestPrj'
        self.prj_PreSum = 'bzhao271828/PreSum'
        # neptune.set_project('bzhao271828/TestPrj')
        self.default_chkpoint_dir = "model_checkpoints/"

    def init_neptune(self, prj_name, exp_name, exp_desc="", tags=["tests"]):
        api_token = "eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vdWkubmVwdHVuZS5haSIsImFwaV91cmwiOiJodHRwczovL3VpLm5lcHR1bmUuYWkiLCJhcGlfa2V5IjoiMzEyMTI3ZmMtYzNjZS00YzVlLTgyMzItODE1MTYzOGM0ZmExIn0="
        if prj_name is None:
            prj_name = self.prj_TestPrj
        self.exp_name = exp_name
        neptune.init(api_token=api_token)
        self.curr_prj = neptune.set_project(project_qualified_name=prj_name)
        # self.curr_prj = neptune.init(project_qualified_name=prj_name, api_token=api_token)
        self.curr_exp = self.curr_prj.create_experiment(name=self.exp_name, description=exp_desc, tags=tags)

    def ckpoint_path(self, chkpoint_name):
        filepath = self.default_chkpoint_dir + chkpoint_name
        return filepath
