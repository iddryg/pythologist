import sys
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
            print(test.comment)

class QCTestGeneric(object):
    def __init__(self,cdf):
        self.cdf = cdf
        self.result,self.comment= self._run()
        return
    def _run(self):
        raise ValueError("Override this with the test")
    @property
    def name(self):
        raise ValueError("Override this")

class QCCheckDBSet(QCTestGeneric):
    @property
    def name(self): return 'Check storage object is set'
    def _run(self):
        if self.cdf.db is None: return('WARNING','h5 storage item for the dataset is not set.')
        return ('PASS','h5 object is set')       

class QCCheckMPPSet(QCTestGeneric):
    @property
    def name(self): return 'Check microns per pixel attribute'
    def _run(self):
        if self.cdf.microns_per_pixel is None: return('WARNING','Microns per pixel is not set.')
        return ('PASS','Microns per pixel is '+str(self.cdf.microns_per_pixel))       

class QCCheckOverlappingFrames(QCTestGeneric):
    @property
    def name(self): return 'Is the same frame name present in multiple samples'
    def _run(self):
        cdf = self.cdf
        return ('PASS','Looks good')    

class QCSampleIDs(QCTestGeneric):
    @property
    def name(self): return 'Is there a 1:1 correspondence between sample_name and sample_id?'
    def _run(self):
        cdf = self.cdf
        df = cdf[['sample_name','sample_id']].drop_duplicates()
        byname = df.groupby('sample_name').count()['sample_id']
        if max(list(byname)) > 1:
            return ('FAIL',"Multiple sample_ids for the same sample_name\n"+str(byname[byname>1]))
        byid = df.groupby('sample_id').count()['sample_name']
        if max(list(byid)) > 1:
            return ('FAIL',"Multiple sample_names for the same sample_id\n"+str(byid[byid>1]))        
        return ('PASS','Looks good')    

class QCFrameIDs(QCTestGeneric):
    @property
    def name(self): return 'Is there a 1:1 correspondence between frame_name and frame_id?'
    def _run(self):
        cdf = self.cdf
        df = cdf[['frame_name','frame_id']].drop_duplicates()
        byname = df.groupby('frame_name').count()['frame_id']
        if max(list(byname)) > 1:
            return ('FAIL',"Multiple frame_ids for the same frame_name\n"+str(byname[byname>1]))
        byid = df.groupby('frame_id').count()['frame_name']
        if max(list(byid)) > 1:
            return ('FAIL',"Multiple frame_names for the same frame_id\n"+str(byid[byid>1]))        
        return ('PASS','Looks good')    
