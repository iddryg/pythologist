import sys, json
from collections import namedtuple

Result = namedtuple('Result',['result','count','total','about'])
class QC(object):
    def __init__(self,cdf,verbose=False):
        self.cdf = cdf
        self.verbose = verbose
        self._test_list = [
           QCMPPSet,
           QCDBSet,
           QCSampleIDs,
           QCFrameIDs,
           QCProjectIDs,
           QCOverlappingFrames,
           QCPhenotypeRules,
           QCPhenotypeConsistency,
           QCScoredNameConsistency,
           QCRegionConsistency,
           QCRegionRules,
           QCRegionSize,
        ]
        self._tests = None
    def run_tests(self):
        # set the _tests property
        self._tests = [x(self.cdf) for x in self._test_list]
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

class QCDBSet(QCTestGeneric):
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


class QCMPPSet(QCTestGeneric):
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

class QCOverlappingFrames(QCTestGeneric):
    @property
    def name(self): return 'Is the same frame name present in multiple samples?'
    def run(self):
        cdf = self.cdf
        frames = cdf[['sample_name','frame_name']].drop_duplicates()
        cnts = frames.groupby('frame_name').count()['sample_name']
        if max(cnts) > 1:
            return Result(result='WARNING',
                          about="frame_name is present in multiple samples. This might indicate duplicated or erroneously named data.\n"+\
                                str(frames[frames['frame_name'].isin(cnts[cnts>1].index)].set_index('sample_name')),
                          count = len(cnts[cnts>1]),
                          total = frames.shape[0]
              )
        return Result(result='PASS',
                      about="frame_name's are all in their own samples",
                      count=0,
                      total=frames.shape[0])    

class QCPhenotypeRules(QCTestGeneric):
    @property
    def name(self): return 'Are the same phenotypes listed and following rules for mutual exclusion?'
    def run(self):
        cdf = self.cdf
        # Check for multiple calls
        psum = cdf.apply(lambda x: sum(x['phenotype_calls'].values())>1,1)
        if cdf.loc[psum].shape[0]>0:
            return Result(result='FAIL',
                          about='There are multiple phenotypes defined a cell. This is a format ERROR, and no more phenotype rules will be checked.  Fix this first. (only showing first 5 indecies): '+str(cdf.loc[psum].head().index.tolist()),
                          count=None,
                          total=None
              )
        if 'phenotype_label' not in cdf.columns:
            return Result(result='WARNING',
                          about='There is no phenotype label column defined so cannot check consistency',
                          count=None,
                          total=None)   
        # Check the zeros
        zeros = cdf.loc[cdf['phenotype_label'].isna()]
        if zeros.shape[0] > 0:
            # we have zeros
            psum = zeros.apply(lambda x: sum(x['phenotype_calls'].values())!=0,1)
            if zeros.loc[psum].shape[0] >0:
                return Result(result='FAIL',
                          about='There are non-zero phenotypes in a null phenotype_label.  This is an error.  Only showing first 5 indecies: '+str(zeros.loc[psum].head().index.tolist()),
                          count=None,
                          total=None
                )
        # Check consistency of phenotype_label
        concordance = cdf.loc[~cdf['phenotype_label'].isna()].apply(lambda x: x['phenotype_calls'][x['phenotype_label']]==1,1)
        if not concordance.all():
            mismatch = cdf.loc[~cdf['phenotype_label'].isna()].loc[~concordance].head()
            return Result(result='FAIL',
                          about='phenotype_label not matching call (only showing first 5 indecies): '+"\n"+str(mismatch.index.tolist()),
                          count=None,
                          total=None
                   )
        return Result(result='PASS',
               about='phenotype_calls and phenotype_label follows expected rules',
               count=None,
               total=None
                   )

class QCPhenotypeConsistency(QCTestGeneric):
    @property
    def name(self): return 'Are the same phenotypes included on all images?'
    def run(self):
        cdf = self.cdf
        phenotypes = set(cdf.phenotypes)
        checks = cdf[['project_name','sample_name','frame_name','phenotype_calls']].copy()
        checks['phenotype_calls'] = checks['phenotype_calls'].apply(lambda x: json.dumps(sorted(list(x.keys()))))
        checks = checks.drop_duplicates()
        checks['phenotype_calls'] = checks['phenotype_calls'].apply(lambda x: json.loads(x))
        issue_total = 0
        issue_count = 0
        pf = 'PASS'
        log = []
        for i,r in checks.iterrows():
            issue_total+=1
            # see which rows have different phenotypes
            calls = set(r['phenotype_calls'])
            if len(phenotypes-calls) > 0:
                issue_count+=1
                pf = 'FAIL'
                log.append([str(r[['project_name','sample_name','frame_name']].tolist())+' is missing or has cells missing phenotype(s) '+str(phenotypes-calls)])
            elif len(calls-phenotypes) > 0:
                issue_count +=1
                pf = 'FAIL'
                log.append([str(r)+' unknown error where phenotype calls has phenotypes that are unknown to the CDF class']) 
        return Result(result=pf,
                      about='Consistent phenotypes' if len(log) == 0 else json.dumps(log,indent=4),
                      count=issue_count,
                      total=issue_total)    
class QCScoredNameConsistency(QCTestGeneric):
    @property
    def name(self): return 'Are the same scored names included on all images?'
    def run(self):
        cdf = self.cdf
        scored_names = set(cdf.scored_names)
        checks = cdf[['project_name','sample_name','frame_name','scored_calls']].copy()
        checks['scored_calls'] = checks['scored_calls'].apply(lambda x: json.dumps(sorted(list(x.keys()))))
        checks = checks.drop_duplicates()
        checks['scored_calls'] = checks['scored_calls'].apply(lambda x: json.loads(x))
        issue_total = 0
        issue_count = 0
        pf = 'PASS'
        log = []
        for i,r in checks.iterrows():
            issue_total+=1
            # see which rows have different scored_names
            calls = set(r['scored_calls'])
            if len(scored_names-calls) > 0:
                issue_count+=1
                pf = 'FAIL'
                log.append([str(r[['project_name','sample_name','frame_name']].tolist())+' is missing or has cells missing scored name(s) '+str(scored_names-calls)])
            elif len(calls-scored_names) > 0:
                issue_count +=1
                pf = 'FAIL'
                log.append([str(r)+' unknown error where scored_calls has scored_name that are unknown to the CDF class']) 
        return Result(result=pf,
                      about='Consistent scored_names' if len(log) == 0 else json.dumps(log,indent=4),
                      count=issue_count,
                      total=issue_total)    
class QCRegionConsistency(QCTestGeneric):
    @property
    def name(self): return 'Are the same regions represented the same with an image and across images?'
    def run(self):
        cdf = self.cdf
        regions = set(cdf.regions)
        checks = cdf[['project_name','sample_name','frame_name','regions']].copy()
        checks['regions'] = checks['regions'].apply(lambda x: json.dumps(sorted(list(x.keys()))))
        checks = checks.drop_duplicates()
        issue_total = 0
        issue_count = 0
        pf = 'PASS'
        log = []
        # see if each image is consistent
        cnts = checks.groupby(['project_name','sample_name','frame_name']).count()['regions']
        issue_total+=1
        if len(cnts[cnts>1]) > 0:
            issue_count += 1
            pf = 'FAIL'
            log.append('there are images with multiple sets of regions defined. each image should have the same set of regions. '+"\n"+str(cnts[cnts>1]))
        # switch back to arrays
        checks['regions'] = checks['regions'].apply(lambda x: json.loads(x))
        for i,r in checks.iterrows():
            issue_total+=1
            # see which rows have different scored_names
            calls = set(r['regions'])
            if len(regions-calls) > 0:
                issue_count+=1
                pf = 'FAIL'
                log.append([str(r[['project_name','sample_name','frame_name']].tolist())+' is missing or has cells missing region(s) '+str(regions-calls)])
            elif len(calls-regions) > 0:
                issue_count +=1
                pf = 'FAIL'
                log.append([str(r)+' unknown error where regions has a region(s) unknown to the CDF class']) 
        return Result(result=pf,
                      about='Consistent regions' if len(log) == 0 else json.dumps(log,indent=4),
                      count=issue_count,
                      total=issue_total)    

class QCRegionSize(QCTestGeneric):
    @property
    def name(self): return 'Do we have any region sizes so small they should consider being excluded?'
    @property
    def minimum_fraction(self): return 0.05
    @property
    def minimum_pixels(self): return 500    
    def run(self):
        cdf = self.cdf
        regions = set(cdf.regions)
        checks = cdf[['project_name','sample_name','frame_name','regions']].copy()
        checks['region_names'] = checks['regions'].apply(lambda x: json.dumps(sorted(list(x.keys()))))
        checks = checks.groupby(['project_name','sample_name','frame_name','region_names']).first().reset_index()
        checks['region_names'] = checks['region_names'].apply(lambda x: json.loads(x))
        checks['total_size'] = checks.apply(lambda x: sum(x['regions'].values()),1)
        issue_total = 2
        issue_count = 0
        pf = 'PASS'
        log1 = []
        for i,r in checks.iterrows():
            # how which regions have regions of size zero
            zero_count = len([y for y in [r['regions'][name] for name in r['region_names']] if y==0])
            if zero_count > 0:
                pf = 'WARNING'
                log1.append('Zero size regions are included in the data.  These are generally fine to have, but they will not contribute to count density calculations for that region. (only showing one example)'+"\n"+str(r[['project_name','sample_name','frame_name','regions']].tolist()))
        if len(log1) > 0: 
          issue_count += 1
          log1 = [log1[0]]
        ## Now look for small regions
        log2 = []
        for i,r in checks.iterrows():
            # how which regions have regions of size zero
            small_count = len([y for y in [r['regions'][name] for name in r['region_names']] if ((y/r['total_size'])<self.minimum_fraction or y < self.minimum_pixels) and y > 0])
            if small_count > 0:
                pf = 'WARNING'
                log2.append('Very small non-zero regions are included in the data'+str(r[['project_name','sample_name','frame_name','regions']].tolist()))
        if len(log2) > 0: issue_count += 1
        return Result(result=pf,
                      about='No zero or very small regions' if len(log1+log2) == 0 else json.dumps(log1+log2,indent=4),
                      count=issue_count,
                      total=issue_total)    

class QCRegionRules(QCTestGeneric):
    @property
    def name(self): return 'Are the same regions listed matching a valid region_label'
    def run(self):
        cdf = self.cdf
        if 'region_label' not in cdf.columns:
            return Result(result='ERROR',
                          about='There is no region label column defined so cannot check consistency',
                          count=None,
                          total=None)   
        # Check consistency of phenotype_label
        concordance = cdf.apply(lambda x: x['region_label'] in x['regions'].keys(),1)
        if not concordance.all():
            mismatch = cdf.loc[~concordance].head()
            return Result(result='FAIL',
                          about='region_label not matching any regions key (only showing first 5 indecies): '+"\n"+str(mismatch.index.tolist()),
                          count=None,
                          total=None
                   )
        return Result(result='PASS',
               about='regions and region_label follows expected rules',
               count=None,
               total=None
                   )  

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
                          count=sum(byname[byname>1]),
                          total=df.shape[0])

        byid = df.groupby('sample_id').count()['sample_name']
        if max(list(byid)) > 1:
            return Result(result='FAIL',
                          about="Multiple sample_names for the same sample_id\n"+str(byid[byid>1]),
                          count=sum(byid[byid>1]),
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
                          count=sum(byname[byname>1]),
                          total=df.shape[0])

        byid = df.groupby('frame_id').count()['frame_name']
        if max(list(byid)) > 1:
            return Result(result='FAIL',
                          about="Multiple frame_names for the same frame_id\n"+str(byid[byid>1]),
                          count=sum(byid[byid>1]),
                          total=df.shape[0])

        return Result(result='PASS',
                          about='Good concordance.',
                          count=0,
                          total=df.shape[0])

class QCProjectIDs(QCTestGeneric):
    @property
    def name(self): return 'Is there a 1:1 correspondence between project_name and project_id?'
    def run(self):
        cdf = self.cdf
        df = cdf[['project_name','project_id']].drop_duplicates()
        byname = df.groupby('project_name').count()['project_id']
        if max(list(byname)) > 1:
            return Result(result='FAIL',
                          about="Multiple project_ids for the same project_name\n"+str(byname[byname>1]),
                          count=sum(byname[byname>1]),
                          total=df.shape[0])

        byid = df.groupby('project_id').count()['project_name']
        if max(list(byid)) > 1:
            return Result(result='FAIL',
                          about="Multiple project_names for the same project_id\n"+str(byid[byid>1]),
                          count=sum(byid[byid>1]),
                          total=df.shape[0])

        return Result(result='PASS',
                          about='Good concordance.',
                          count=0,
                          total=df.shape[0])
