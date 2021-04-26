
from collections import namedtuple
import pandas as pd
import numpy as np
import sys

Result = namedtuple('Result',['result','count','total','about'])
class QC(object):
    def __init__(self,proj,verbose=False):
        self.proj = proj
        self.verbose = verbose
        self._test_list = []
        self._tests = None
    def run_tests(self):
        # set the _tests property
        self._tests = [x(self.proj) for x in self._test_list]
    def print_results(self):
        if self._tests is None:
            if self.verbose: sys.stderr.write("tests is None so running tests\n")
            self.run_tests()
        for test in self._tests:
            print('==========')
            print(test.name)
            print(test.result)
            print(test.about)
            if test.total is not None: print('Issue count: '+str(test.count)+'/'+str(test.total))

    def channel_histograms(self,minvalue=None,maxvalue=None,bins=100):
        def _get_image_intensities(proj,sample_id,frame_id,image_id,processed_image_id):
            ci = proj.get_image(sample_id,frame_id,image_id)
            pi = proj.get_image(sample_id,frame_id,processed_image_id)
            ciflat = ci.flatten()
            piflat = pi.flatten().astype(bool)
            return ciflat[piflat]
        def _get_image_max(*args):
            return _get_image_intensities(*args).max()
        def _get_image_min(*args):
            return _get_image_intensities(*args).min()
        # begin logic
        test = self.proj.channel_image_dataframe
        if maxvalue is None:
            if self.verbose: sys.stderr.write("calculating max value\n")
            maxvalue = test.apply(lambda x: _get_image_max(self.proj,
                                                             x['sample_id'],
                                                             x['frame_id'],
                                                             x['image_id'],
                                                             x['processed_image_id']),1).max()
        if minvalue is None:
            if self.verbose: sys.stderr.write("calculating min value\n")
            minvalue = test.apply(lambda x: _get_image_min(self.proj,
                                                             x['sample_id'],
                                                             x['frame_id'],
                                                             x['image_id'],
                                                             x['processed_image_id']),1).min()
        step = (maxvalue-minvalue)/bins
        bins = [step*x for x in range(0,bins)]
        cnames = ['project_id','project_name','sample_id','sample_name','frame_id','frame_name','channel_label','channel_abbreviation']
        if self.verbose: sys.stderr.write("calculating histogram values\n")
        test2 = test.apply(lambda x: 
            pd.Series(dict(zip(
                cnames +\
                ['counts','bins'],
                [x['project_id'],x['project_name'],x['sample_id'],x['sample_name'],x['frame_id'],x['frame_name'],x['channel_label'],x['channel_abbreviation']]+\
                list(np.histogram(_get_image_intensities(self.proj,
                                        x['sample_id'],
                                        x['frame_id'],
                                        x['image_id'],
                                        x['processed_image_id']),bins))
            )))
            ,1)
        test3 = []
        if self.verbose: sys.stderr.write("building long dataframe of histogram values\n")
        for index,r in test2.set_index(cnames).iterrows():
            df = pd.DataFrame({'counts':r['counts'],'bins':r['bins'][:-1]})
            for i,name in enumerate(cnames):
                df[name] = index[i]
            test3.append(df)
        test3 = pd.concat(test3)
        return test3


class QCTestGeneric(object):
    def __init__(self,proj):
        self.proj = proj
        self._result = None
        return
    def run(self):
        raise ValueError("Override this with the test")
    @property
    def name(self):
        raise ValueError("Override this")
    @property
    def result(self):
        if self._result is None: self._result = self.run()
        return self._result.result
    @property
    def about(self):
        if self._result is None: self._result = self.run()
        return self._result.about    
    @property
    def count(self):
        if self._result is None: self._result = self.run()
        return self._result.count    
    @property
    def total(self):
        if self._result is None: self._result = self.run()
        return self._result.total 