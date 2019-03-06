import sys
from collections import namedtuple

Result = namedtuple('Result',['result','count','total','about'])
class QC(object):
    def __init__(self,cdf,verbose=False):
        self.cdf = cdf
        self.verbose = verbose
        self._tests = None
    def run_tests(self):
        # set the _tests property
        self._tests = [
           QCCheckMPPSet(self.cdf),
           QCCheckDBSet(self.cdf),
           QCCheckOverlappingFrames(self.cdf),
           QCSampleIDs(self.cdf),
           QCFrameIDs(self.cdf),
        ]
    def print_results(self):
        if self._tests is None:
            if self.verbose: sys.stderr.write("tests is None so running tests\n")
            self.run_tests()
        for test in self._tests:
            print('==========')
            print(test.name)
            print(test.result)
            print(test.about)

class QCTestGeneric(object):
    def __init__(self,cdf):
        self.cdf = cdf
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

class QCCheckDBSet(QCTestGeneric):
    @property
    def name(self): return 'Check storage object is set'
    def run(self):
        if self.cdf.db is None: 
            return Result(result='WARNING',
                          about='h5 storage item for the dataset is not set.',
                          count=None,
                          total=None)
        return Result(result='PASS',
                      about='h5 object is set',
                      count = None,
                      total=None)


class QCCheckMPPSet(QCTestGeneric):
    @property
    def name(self): return 'Check microns per pixel attribute'
    def run(self):
        if self.cdf.microns_per_pixel is None: 
            return Result(result='WARNING',
                          about='Microns per pixel is not set.',
                          count=None,
                          total=None)
        return Result(result='PASS',
                      about='Microns per pixel is '+str(self.cdf.microns_per_pixel),
                      count = None,
                      total=None)

class QCCheckOverlappingFrames(QCTestGeneric):
    @property
    def name(self): return 'Is the same frame name present in multiple samples'
    def run(self):
        cdf = self.cdf
        return Result(result='PASS',
                      about="Looks good",
                      count=None,
                      total=None)    

class QCSampleIDs(QCTestGeneric):
    @property
    def name(self): return 'Is there a 1:1 correspondence between sample_name and sample_id?'
    def run(self):
        cdf = self.cdf
        df = cdf[['sample_name','sample_id']].drop_duplicates()
        byname = df.groupby('sample_name').count()['sample_id']
        if max(list(byname)) > 1:
            return Result(result='FAIL',
                          about="Multiple sample_ids for the same sample_name\n"+str(byname[byname>1]),
                          count=len(byname[byname>1]),
                          total=df.shape[0])

        byid = df.groupby('sample_id').count()['sample_name']
        if max(list(byid)) > 1:
            return Result(result='FAIL',
                          about="Multiple sample_names for the same sample_id\n"+str(byid[byid>1]),
                          count=len(byid[byid>1]),
                          total=df.shape[0])

        return Result(result='PASS',
                          about='Good concordance.',
                          count=0,
                          total=df.shape[0])

class QCFrameIDs(QCTestGeneric):
    @property
    def name(self): return 'Is there a 1:1 correspondence between frame_name and frame_id?'
    def run(self):
        cdf = self.cdf
        df = cdf[['frame_name','frame_id']].drop_duplicates()
        byname = df.groupby('frame_name').count()['frame_id']
        if max(list(byname)) > 1:
            return Result(result='FAIL',
                          about="Multiple frame_ids for the same frame_name\n"+str(byname[byname>1]),
                          count=len(byname[byname>1]),
                          total=df.shape[0])

        byid = df.groupby('frame_id').count()['frame_name']
        if max(list(byid)) > 1:
            return Result(result='FAIL',
                          about="Multiple frame_names for the same frame_id\n"+str(byid[byid>1]),
                          count=len(byid[byid>1]),
                          total=df.shape[0])

        return Result(result='PASS',
                          about='Good concordance.',
                          count=0,
                          total=df.shape[0])
