""" These modules are to facilitate reading  PerkinElmer inForm outputs into python

**Example:** reading a group of folders

.. code-block:: python

    r = read.software()

"""
import os, re, sys
from collections import OrderedDict
import pandas as pd
import numpy as np

class CellFrame(pd.DataFrame):
    _metadata = ['_thresholds','_components','_scores','_continuous']
    @property
    def _constructor(self):
        return CellFrame
    @property
    def _constructor_sliced(self):
        return CellFrame
    def remove_threshold(self,phenotype):
        """ Wipe the current thresholding from a phenotype"""
        v = pd.DataFrame(self).copy()
        v.loc[v['phenotype']==phenotype,'threshold_marker'] = np.nan
        v.loc[v['phenotype']==phenotype,'threshold_call'] = np.nan
        v.loc[v['phenotype']==phenotype,'full_phenotype'] = v.loc[v['phenotype']==phenotype,'phenotype']
        v = CellFrame(v)
        v._thresholds = self._thresholds.copy()
        v._components = self._components.copy()
        v._scores = self._scores.copy()
        v._continuous = self._continuous.copy()
        return v

    def add_threshold(self,phenotype,component,name):
        v = pd.DataFrame(self).copy()
        v['dbid'] = range(1,v.shape[0]+1,1)
        mythresh = self.thresholds[self.thresholds['component']==component]
        combo = v.merge(mythresh,on=['sample','frame'])
        choice = pd.DataFrame(combo[['compartment','component']].iloc[0]).apply(lambda x: x[0]+' '+x[1]+' Mean (Normalized Counts, Total Weighting)')[0]
        combo = combo[['dbid','sample','frame','id','phenotype','threshold',choice]].\
            rename(columns={choice:'observed'})
        low = combo[(combo['threshold']>combo['observed'])&(combo['phenotype']==phenotype)].set_index('dbid')
        high = combo[(combo['threshold']<=combo['observed'])&(combo['phenotype']==phenotype)].set_index('dbid')
        v = v.set_index('dbid')
        v.loc[low.index,'threshold_marker'] = name
        v.loc[low.index,'threshold_call'] = '-'
        v.loc[high.index,'threshold_marker'] = name
        v.loc[high.index,'threshold_call'] = '+'
        v = v.reset_index(drop=True)
        v.loc[v['phenotype']==phenotype,'full_phenotype'] = v[v['phenotype']==phenotype][['phenotype','threshold_marker','threshold_call']].\
            dropna().apply(lambda x: x[0]+' '+x[1]+x[2],1)
        v = CellFrame(v)
        v._thresholds = self._thresholds.copy()
        v._components = self._components.copy()
        v._scores = self._scores.copy()
        v._continuous = self._continuous.copy()
        return v

    def collapse_phenotypes(self,input_names,output_name):
        """ Collapse a list of phenotypes into another name, also removes thresholding """
        v = self.copy()
        for input_name in input_names: v = v.remove_threshold(input_name)
        v = v.remove_threshold(output_name)
        v = pd.DataFrame(v.copy())
        v.loc[v['phenotype'].isin(input_names),'full_phenotype'] = output_name
        v.loc[v['phenotype'].isin(input_names),'phenotype'] = output_name

        v = CellFrame(v)
        v._thresholds = self._thresholds.copy()
        v._components = self._components.copy()
        v._scores = self._scores.copy()
        v._continuous = self._continuous.copy()
        return v
    def add_continuous(self,compartment,component,name):
        if name in self.columns:
            raise ValueError("ERROR name "+name+" already is in early use")
        if name in self.df.columns:
            raise ValueError("ERROR name "+name+" already is in use")
        v = compartment+' '+component+' Mean (Normalized Counts, Total Weighting)'
        if v not in self.columns:
            raise ValueError("ERROR "+v+" not in columns")
        c = self.copy()
        c._thresholds = self._thresholds.copy()
        c._components = self._components.copy()
        c._scores = self._scores.copy()
        c._continuous = self._continuous.copy()
        c._continuous[v] = name
        return c
    def remove_continuous(self,name):
        c = self.copy()
        c._thresholds = self._thresholds.copy()
        c._components = self._components.copy()
        c._scores = self._scores.copy()
        c._continuous = self._continuous.copy()
        for n1 in c._continuous:
            if c._continuous[n1] == name:
                del c._continuous[n1]
        return c
    @property
    def df(self):
        c = pd.DataFrame(self).copy() 
        keepers = ['sample','frame','id','phenotype','threshold_marker','threshold_call','full_phenotype','x','y','area']
        for n1 in self._continuous:
            c = c.rename(columns={n1:self._continuous[n1]})
        keepers += list(self._continuous.values())
        return c[keepers]
    @staticmethod
    def read_inForm(path,verbose=False):
        """ path is the location of the folds
            """ 
        return _SampleSet(path,verbose).cells
    @property
    def samples(self):
        return self._samples
    @property
    def components(self): return self._components
    @property
    def thresholds(self): return self._thresholds
    @property
    def scores(self): return self._scores

class GenericSample:
    @property
    def scores(self):
        v = []
        for sample in self.samples:
            for frame in self.samples[sample]:
                f = self.samples[sample][frame].frame_scores
                v.append(f)
        return pd.concat(v)
    @property
    def thresholds(self):
        v = []
        for sample in self.samples:
            for frame in self.samples[sample]:
                f = self.samples[sample][frame].frame_stains
                for component in f:
                    v.append([sample,frame,component,f[component]['compartment'],f[component]['threshold']])
        v = pd.DataFrame(v)
        v.columns = ['sample','frame','component','compartment','threshold']
        return v
    @property
    def components(self):
        clist = []
        for sample in self.samples:
            for frame in self.samples[sample]:
                f = self.samples[sample][frame]
                clist.append(f.frame_components)
        if not all(sorted(x)==sorted(clist[0]) for x in clist):
            sys.stderr.write("WARNING: not all samples have the same components.\n")
        return clist[0]


class _SampleSet(GenericSample):
    """ Read in a Folder containing sample folders recursively

    .. note:: This connection class is the primary portal to
              work with the REDCap system

    :param path: sample folder or folder containing sample folders
    :type path: string
    """
    def __init__(self,path,verbose=False):
        GenericSample.__init__(self)
        self._path = path
        self._samples = OrderedDict()
        for p, dirs, files in os.walk(self._path,followlinks=True,topdown=False):
            segs = [x for x in files if re.search('_cell_seg_data.txt$',x)]
            if len(segs) == 0: continue
            s = _Sample(p,verbose)
            self._samples[s.name] = s

        # update our samples layout
        s = OrderedDict()
        for samp in self._samples: s[samp] = self._samples[samp].samples[samp]
        self._samples = s
    @property
    def samples(self): return self._samples
    @property
    def cells(self):
        rows = []
        for sample in self._samples:
            for frame in self._samples[sample]:
                rows.append(self._samples[sample][frame].cells)
        c = CellFrame(pd.concat(rows))
        c._thresholds = self.thresholds
        c._components = self.components
        c._scores = self.scores
        c._continuous = OrderedDict()
        return c


class _Sample(GenericSample):
    def __init__(self,path,verbose=False):
        GenericSample.__init__(self)
        self._path = path
        files = os.listdir(path)
        # Find frames in the same by a filename type we should have
        segs = [x for x in files if re.search('_cell_seg_data.txt$',x)]
        sample_folder = os.path.basename(path)
        self._frames = OrderedDict()
        snames = set()
        for file in segs:
            m = re.match('(.*)_(\[\d+,\d+\])_cell_seg_data.txt$',file)
            sample = m.group(1)
            snames.add(sample)
            frame = m.group(2)
            data = os.path.join(path,file)
            summary = os.path.join(path,sample+'_'+frame+'_cell_seg_data_summary.txt')
            if not os.path.exists(summary):
            	if verbose: sys.stderr.write('Missing summary file '+summary+"\n")
            	summary = None
            score = os.path.join(path,sample+'_'+frame+'_score_data.txt')
            if not os.path.exists(score):
            	raise ValueError('Missing score file '+score)
            f = _Frame(path,sample,frame,data,score,summary)
            self._frames[frame] = f
        if len(snames) > 1:
        	raise ValueError('Error multiple samples in folder '+path)
        self._sample_name = list(snames)[0]
    @property
    def name(self): return self._sample_name
    @property
    def frames(self): return self._frames

    @property
    def samples(self):
        return OrderedDict({self._sample_name:self.frames})

class _Frame(GenericSample):
    def __init__(self,path,sample,frame,seg_file,score_file,summary_file):
        #self._seg = pd.read_csv(seg_file,sep=",")
        GenericSample.__init__(self)
        self._frame = frame
        self._sample = sample
        self._seg = pd.read_csv(seg_file,"\t")
        self._score = OrderedDict(pd.read_csv(score_file,"\t").iloc[0].to_dict())
        checks = ['First','Second','Third']
        # get the stains and thresholds
        self._stains = OrderedDict()
        for check in checks:
            compartment = check+' Cell Compartment'
            stain = check+' Stain Component'
            if compartment in self._score:
                self._stains[self._score[stain]] = OrderedDict({'compartment':self._score[compartment],
                	'threshold':self.frame_thresholds[self._score[stain]]})
        # get the components
        self._components = []
        for name in self._seg.columns:
        	m = re.match('Entire Cell (.*) Mean \(Normalized Counts, Total Weighting\)',name)
        	if not m: continue
        	component = m.group(1)
        	if component not in self._components: self._components.append(component)
    @property
    def frame_stains(self): return self._stains
    @property
    def frame_components(self): return self._components
    @property
    def frame_thresholds(self):
        t = OrderedDict()
        v = [x for x in self._score.keys() if re.search(' Threshold$',x)]
        for entry in v:
            name = re.match('(.*) Threshold$',entry).group(1)
            value = self._score[entry]
            t[name] = value
        return t
    @property
    def frame_scores(self):
        v = self._score.copy()
        v['sample'] = self._sample
        v['frame'] = self._frame
        return pd.DataFrame(pd.Series(v)).transpose()
    @property
    def cells(self):
        keepers = ['Cell ID','Phenotype',
            'Cell X Position',
            'Cell Y Position',
            'Entire Cell Area (pixels)']
        keepers2 = [x for x in self._seg.columns if re.search('Mean \(Normalized Counts, Total Weighting\)$',x)]
        v = self._seg[keepers+keepers2]
        v = v.rename(columns = {'Cell ID':'id',
            'Entire Cell Area (pixels)':'area',
            'Cell X Position':'x',
            'Cell Y Position':'y',
            'Phenotype':'phenotype'})
        v['frame'] = self._frame
        v['sample'] = self._sample
        v['threshold_marker'] = np.nan
        v['threshold_call'] = np.nan
        v['full_phenotype'] = v['phenotype']
        c = CellFrame(v)
        c._thresholds = self.thresholds
        c._components = self.components
        c._scores = self.frame_scores
        c._continuous = OrderedDict()
        return c
    @property
    def samples(self):
        return OrderedDict({self._sample:OrderedDict({self._frame:self})})
