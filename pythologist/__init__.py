""" These modules are to facilitate reading  PerkinElmer inForm outputs into python

**Example:** reading a group of folders

.. code-block:: python

    r = read.software()

"""
import os, re, sys, h5py, json, math
from collections import OrderedDict
import pandas as pd
import numpy as np
import pythologist.spatial

def read_Vectra(path,mpp=0.496,verbose=False,limit=None):
    return InFormCellFrame.read_Vectra(path,mpp,verbose,limit)

def _swap(current,phenotypes,name):
    out = []
    for p in json.loads(current):
        if not p in phenotypes:
            out.append(p)
            continue
        out.append(name)
    return json.dumps(out)
def _add(current,phenotype,name):
    out = []
    for p in json.loads(current):
        if p != phenotype:
            out.append(p)
            continue
        out.append(phenotype+" "+name+"+")
        out.append(phenotype+" "+name+"-")
    return json.dumps(out)
def _swap_tissue(areas,old_name,new_name):
    areas = json.loads(areas)
    v = {}
    for k in list(areas.keys()):
        if k == old_name:
            v[new_name] = areas[k]
        else:
            v[k] = areas[k]
    return json.dumps(v)

class InFormCellFrame(pd.DataFrame):
    ##_default_mpp = 0.496 # microns per pixel on vetra
    #### Things for extending pd.DataFrame #####
    _metadata = ['_thresholds','_components','_scores','_continuous','mpp']
    @property
    def _constructor(self):
        return InFormCellFrame
    def __init__(self,*args,**kw):
        # hide mpp for now when we send to parent since it doesn't use that argument
        kwcopy = kw.copy()
        if 'mpp' in kwcopy: del kwcopy['mpp']
        super(InFormCellFrame,self).__init__(*args,**kwcopy)
        if 'mpp' in kw: self._mpp = kw['mpp']
        else: self._mpp = None
        #else: self._mpp = InFormCellFrame._default_mpp 
        self['frame_stains'] = pd.Series(self['frame_stains'],dtype="category")
        self['tissues_present'] = pd.Series(self['tissues_present'],dtype="category")
        self['phenotypes_present'] = pd.Series(self['phenotypes_present'],dtype="category")
    def __repr__(self): return 'pig'
    def _repr_html_(self): return pd.DataFrame(self)._repr_html_()
    def copy(self):
        v = InFormCellFrame(pd.DataFrame(self).copy(),mpp=self.mpp)
        return v
    def to_hdf(self,path):
        string_version = self.copy()
        string_version['compartment_values'] = string_version.apply(lambda x: json.dumps(x['compartment_values']),1)
        string_version['entire_cell_values'] = string_version.apply(lambda x: json.dumps(x['entire_cell_values']),1)
        pd.DataFrame(string_version).to_hdf(path,'self',
                                  format='table',
                                  mode='w',
                                  complib='zlib',
                                  complevel=9)
        ho = h5py.File(path,'r+')
        o3 = json.dumps(self.mpp)
        ho.attrs['mpp'] = o3
        ho.flush()
        ho.close()
    #### Static methods to read data
    @classmethod
    def read_hdf(cls,path):
        df = pd.read_hdf(path,'self')
        df['compartment_values'] = df.apply(lambda x: json.loads(x['compartment_values']),1)
        df['entire_cell_values'] = df.apply(lambda x: json.loads(x['entire_cell_values']),1)
        seed = cls(df)
        seed.set_mpp(json.loads(h5py.File(path,'r').attrs['mpp']))
        return seed
    @staticmethod
    def read_Vectra(path,mpp=0.496,verbose=False,limit=None):
        """ path is the location of the folds
            """ 
        print(mpp)
        print(verbose)
        print(limit)
        return InFormCellFrame(SampleSet(path,verbose,limit).cells,mpp=mpp)

    #### Properties of the InFormCellFrame
    @property
    def samples(self):
        return None
    @property
    def frame_data(self):
        frame_general = self.df[['folder','sample','frame','total_area','tissues_present','phenotypes_present','frame_stains']].drop_duplicates()
        frame_counts = self.df.groupby(['folder','sample','frame']).\
            count()[['id']].reset_index().rename(columns={'id':'cell_count'})
        frame_data = frame_general.merge(frame_counts,on=['folder','sample','frame'])
        frame_data['tissues_present'] = frame_data.apply(lambda x: json.loads(x['tissues_present']),1)
        frame_data['phenotypes_present'] = frame_data.apply(lambda x: json.loads(x['phenotypes_present']),1)
        frame_data['frame_stains'] = frame_data.apply(lambda x: json.loads(x['frame_stains']),1)
        return frame_data.sort_values(['folder','sample','frame'])
    @property
    def score_data(self):
        rows = []
        for row in self.df[['folder','sample','frame','tissue','tissue_area','total_area','frame_stains']].drop_duplicates().itertuples(index=False):
            stains = json.loads(row.frame_stains)
            for tissue in stains:
                for stain in stains[tissue]:
                    df = pd.Series(stains[tissue][stain])
                    rows.append(pd.Series(row,index=row._fields).append(pd.Series({'stain':stain})).append(df))
        df = pd.DataFrame(rows).drop(columns=['frame_stains']).sort_values(['folder','sample','frame','tissue'])
        return df
    def frame_stains_to_dict(self):
        s = OrderedDict()
        for stain_str in self['frame_stains'].cat.categories:
            s[stain_str] = json.loads(stain_str)
        return s
    def tissues_present_to_dict(self):
        s = OrderedDict()
        for area_str in self['tissues_present'].cat.categories:
            s[area_str] = json.loads(area_str)
        return s
    @property
    def stains(self):
        stains = set()
        sdict = self.frame_stains_to_dict()
        for stain_str in sdict:
            for tissue in sdict[stain_str]:
                for stain in sdict[stain_str][tissue]:
                    stains.add(stain)
        return sorted(list(stains))

    #### Operations to QC an InFormCellFrame
    def rename_tissue(self,old_name,new_name):
        df = self.df.copy()
        df.loc[df['tissue']==old_name,'tissue'] = new_name
        sdict = self.frame_stains_to_dict()
        for jstr in sdict:
            if old_name not in sdict[jstr]:
                sdict[jstr] = jstr
                continue
            new_sdict = OrderedDict()
            for tissue in sdict[jstr]:
                if tissue == old_name: 
                    tissue = new_name
                    if new_name in sdict[jstr]: raise ValueError('cant rename to a tissue that already has been used')
                    new_sdict[tissue] = sdict[jstr][old_name]
                else:
                    new_sdict[tissue] = sdict[jstr][tissue]
            sdict[jstr] = json.dumps(new_sdict)
        df['frame_stains'] = df.apply(lambda x: sdict[x['frame_stains']],1)
        adict = self.tissues_present_to_dict()
        anew = OrderedDict()
        for astr in adict:
            if old_name not in adict[astr]:
                anew[astr] = astr
                continue
            if new_name in adict[astr]: raise ValueError("error can't rename a tissue to one thats already there")
            v = []
            for tissue in adict[astr]:
                if tissue != old_name:
                    v.append(tissue)
                    continue
                else:
                    v.append(new_name)
            anew[astr] = json.dumps(v)
        df['tissues_present'] = df.apply(lambda x: anew[x['tissues_present']],1)
        return InFormCellFrame(df,mpp=self.mpp)

    def collapse_phenotypes(self,input_names,output_name):
        """ Collapse a list of phenotypes into another name, also removes thresholding """
        v = self.df.copy()
        v.loc[v['phenotype'].isin(input_names),'phenotype'] = output_name
        return InFormCellFrame(v,mpp=self.mpp)
    def threshold(self,stain,phenotype,abbrev):
        pheno = self.df[self['phenotype']==phenotype]
        mf = pheno[['id','phenotype','x','y','folder','sample','frame','tissue','compartment_values','entire_cell_values']].\
            merge(self.score_data,on=['folder','sample','frame','tissue'])
        mf = mf[mf['stain']==stain].copy()
        mf['gate'] = '-'
        mf.loc[mf.apply(lambda x: x['compartment_values'][stain][x['compartment']] > x['threshold'],1),'gate'] = '+'
        mf = mf[['folder','sample','frame','phenotype','gate','id']].\
            set_index(['folder','sample','frame','id']).\
            apply(lambda x: x['phenotype']+' '+abbrev+x['gate'],1).reset_index().\
            rename(columns={0:'new_phenotype'})
        mf2 = self.df.merge(mf,on=['folder','sample','frame','id'],how='left')
        to_change = mf2['new_phenotype'].notna()
        mf2.loc[to_change,'phenotype'] = mf2.loc[to_change,'new_phenotype']
        return InFormCellFrame(mf2.drop(columns=['new_phenotype']),mpp=self.mpp)

    def kNearestNeighborsCross(cf,phenotypes,k=1,threads=1,dlim=None):
        nn = pythologist.spatial.kNearestNeighborsCross(cf,phenotypes,k=k,threads=threads,dlim=dlim)
        return pythologist.spatial.CellFrameNearestNeighbors(cf,nn)
    def set_mpp(self,value):
        self._mpp = value
    @property
    def mpp(self): return self._mpp
    #    if hasattr(self,'_mpp'): return self._mpp
    #    else: return None
    #    self._mpp = InFormCellFrame._default_mpp
    #    return self._mpp

    #@property
    #def _constructor_sliced(self):
    #    return InFormCellFrame
    def _remove_threshold(self,phenotype):
        #raise ValueError("Haven't made compatible with the counts and phenotypes_present")
        """ Wipe the current thresholding from a phenotype"""
        v = pd.DataFrame(self).copy()
        v.loc[v['phenotype']==phenotype,'threshold_marker'] = np.nan
        v.loc[v['phenotype']==phenotype,'threshold_call'] = np.nan
        v.loc[v['phenotype']==phenotype,'full_phenotype'] = v.loc[v['phenotype']==phenotype,'phenotype']
        v = InFormCellFrame(v,mpp=self.mpp)
        v._thresholds = self._thresholds.copy()
        v._components = self._components.copy()
        v._scores = self._scores.copy()
        v._continuous = self._continuous.copy()
        v.set_mpp(self.mpp)
        return v

    def add_threshold(self,phenotype,component,name):
        v = pd.DataFrame(self).copy()
        v['dbid'] = range(1,v.shape[0]+1,1)
        mythresh = self.thresholds[self.thresholds['component']==component]
        combo = v.merge(mythresh,on=['sample','frame'])
        #print(combo)
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
        v['phenotypes_present'] = v.apply(lambda x: _add(x['phenotypes_present'],phenotype,name),1)
        v = InFormCellFrame(v,mpp=self.mpp)
        v._thresholds = self._thresholds.copy()
        v._components = self._components.copy()
        v._scores = self._scores.copy()
        v._continuous = self._continuous.copy()
        v.set_mpp(self.mpp)
        # fix those phenotypes present
        return v




    def lock_all_thresholds(self):
        v = pd.DataFrame(self).copy()
        v['phenotype'] = v['full_phenotype'].copy()
        v['threshold_marker'] = np.nan
        v['threshold_call'] = np.nan
        v = InFormCellFrame(v,mpp=self.mpp)
        v._thresholds = self._thresholds.copy()
        v._components = self._components.copy()
        v._scores = self._scores.copy()
        v._continuous = self._continuous.copy()
        v.set_mpp(self.mpp)
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
        c.set_mpp(self.mpp)
        c._continuous[v] = name
        return c

    def remove_continuous(self,name):
        c = self.copy()
        c._thresholds = self._thresholds.copy()
        c._components = self._components.copy()
        c._scores = self._scores.copy()
        c._continuous = self._continuous.copy()
        c.set_mpp(self.mpp)
        for n1 in c._continuous:
            if c._continuous[n1] == name:
                del c._continuous[n1]
        return c

    @property
    def df(self):
        return pd.DataFrame(self).copy()
    @property
    def components(self): return self._components
    @property
    def thresholds(self): return self._thresholds
    @property
    def scores(self): return self._scores
    @property
    def frame_counts(self):
        # Assuming all phenotypes and all tissues could be present in all frames
        basic = self.df.groupby(['sample','frame','tissue','full_phenotype']).first().reset_index()[['sample','frame','tissue','tissue_area','total_area','phenotype','threshold_marker','threshold_call','full_phenotype']]

        cnts = self.df.groupby(['sample','frame','tissue','full_phenotype']).count().\
             reset_index()[['sample','frame','tissue','full_phenotype','id']].\
             rename(columns ={'id':'count'})
        cnts = cnts.merge(basic,on=['sample','frame','tissue','full_phenotype'])
        #return cnts
        df = pd.DataFrame(self)[['sample','frame','phenotypes_present','tissues_present']].groupby(['sample','frame']).first().reset_index()
        empty = []
        sample_tissues = OrderedDict()
        sample_phenotypes = OrderedDict()
        all_tissues = set()
        all_phenotypes = set()
        for frame in df.itertuples():
            areas = json.loads(getattr(frame,"tissues_present"))
            phenotypes = json.loads(getattr(frame,"phenotypes_present"))
            sname = getattr(frame,"sample")
            fname = getattr(frame,"frame")
            tlist = set()
            plist = set()
            for tissue in areas.keys():
                if tissue == 'All': continue
                tlist.add(tissue)
                all_tissues.add(tissue)
                total = areas['All']
                for phenotype in phenotypes:
                    plist.add(phenotype)
                    all_phenotypes.add(phenotype)
            if sname not in sample_tissues: sample_tissues[sname] = set()
            sample_tissues[sname] |= tlist
            if sname not in sample_phenotypes: sample_phenotypes[sname] = set()
            sample_phenotypes[sname] |= plist
        thresh = {}
        for frame in cnts.itertuples():
            pheno = getattr(frame,"phenotype")
            full = getattr(frame,"full_phenotype")
            label = getattr(frame,"threshold_marker")
            call = getattr(frame,"threshold_call")
            if not isinstance(label,str): continue
            thresh[full] = {'label':label,'call':call,'phenotype':pheno}
        for frame in df.itertuples():
            sname = getattr(frame,"sample")
            fname = getattr(frame,"frame")
            tlist = list(sample_tissues[sname])
            plist = list(sample_phenotypes[sname])
            #tlist = list(all_tissues)
            #plist = list(all_phenotypes)
            for tissue in sorted(tlist):
                for phenotype in sorted(plist):            
                    sub = cnts[(cnts['sample']==sname)&(cnts['frame']==fname)&(cnts['tissue']==tissue)&(cnts['full_phenotype']==phenotype)]
                    subcnt = sub.shape[0]
                    if subcnt != 0: continue
                    g = pd.Series({'sample':sname,
                                   'frame':fname,
                                   'tissue':tissue,
                                   'tissue_area':np.nan if tissue not in areas else areas[tissue],
                                   'total_area':total,
                                   'full_phenotype':phenotype,
                                   'threshold_marker': np.nan if phenotype not in thresh else thresh[phenotype]['label'],
                                   'threshold_call': np.nan if phenotype not in thresh else thresh[phenotype]['call'],
                                   'phenotype': np.nan if phenotype not in thresh else thresh[phenotype]['phenotype'],
                                   'count':0})
                    empty.append(g)

        out = pd.concat([cnts,pd.DataFrame(empty)])\
              [['sample','frame','tissue','phenotype','threshold_marker','threshold_call','full_phenotype','tissue_area','total_area','count']].\
            sort_values(['sample','frame','tissue','full_phenotype'])
        out['tissue_area'] = out['tissue_area'].fillna(0)
        out['density'] = out.apply(lambda x: np.nan if float(x['tissue_area']) == 0 else float(x['count'])/float(x['tissue_area']),1)
        out['density_um2'] = out.apply(lambda x: x['density']/(self.mpp*self.mpp),1)
        out['tissue_area_um2'] = out.apply(lambda x: x['tissue_area']/(self.mpp*self.mpp),1)
        out['total_area_um2'] = out.apply(lambda x: x['total_area']/(self.mpp*self.mpp),1)
        return(out)
    @property
    def sample_counts(self):
        fc = self.frame_counts

        v = fc[fc['density'].notnull()].groupby(['sample','tissue','full_phenotype']).\
            count().reset_index()[['sample','tissue','full_phenotype','density']].\
            rename(columns={'density':'present_count'})
        basic = fc.groupby(['sample','tissue','full_phenotype']).first().reset_index()[['sample','tissue','phenotype','threshold_marker','threshold_call','full_phenotype']]
        mean = fc.groupby(['sample','tissue','full_phenotype']).mean().reset_index()[['sample','tissue','full_phenotype','density']].rename(columns={'density':'mean'})
        std = fc.groupby(['sample','tissue','full_phenotype']).std().reset_index()[['sample','tissue','full_phenotype','density']].rename(columns={'density':'std_dev'})
        cnt =  fc.groupby(['sample','tissue','full_phenotype']).count().reset_index()[['sample','tissue','full_phenotype','count']].rename(columns={'count':'frame_count'})
        out = basic.merge(mean,on=['sample','tissue','full_phenotype']).merge(std,on=['sample','tissue','full_phenotype']).merge(cnt,on=['sample','tissue','full_phenotype'])
        out = out.merge(v,on=['sample','tissue','full_phenotype'],how='left')
        out['present_count'] = out['present_count'].fillna(0).astype(int)
        out['std_err'] = out.apply(lambda x: np.nan if x['present_count'] == 0 else x['std_dev']/(math.sqrt(x['present_count'])),1)
        out['mean_um2'] = out.apply(lambda x: x['mean']/(self.mpp*self.mpp),1)
        out['std_dev_um2'] = out.apply(lambda x: x['std_dev']/(self.mpp*self.mpp),1)
        out['std_err_um2'] = out.apply(lambda x: x['std_err']/(self.mpp*self.mpp),1)
        return out

class GenericSamples:
    @property
    def cells():
        # should be overridden by child
        return None


class SampleSet(GenericSamples):
    """ Read in a Folder containing sample folders recursively

    .. note:: This connection class is the primary portal to
              work with the REDCap system

    :param path: sample folder or folder containing sample folders
    :type path: string
    """
    def __init__(self,path,verbose=False,limit=None):
        base = os.path.abspath(path)
        path = os.path.abspath(path)
        GenericSamples.__init__(self)
        self._path = path
        self._cells = None
        self._samples = []
        z = 0
        for p, dirs, files in os.walk(self._path,followlinks=True,topdown=False):
            mydir = p[len(path):]
            z += 1
            segs = [x for x in files if re.search('_cell_seg_data.txt$',x)]
            if len(segs) == 0: continue
            s = Sample(p,mydir,verbose)
            self._samples.append(s)
            if limit is not None and z >= limit: break
        # update our samples layout
        #s = OrderedDict()
        #or samp in self._samples: s[samp] = self._samples[samp].samples[samp]
        #self._samples = s
    @property
    def cells(self): 
        if self._cells is not None: return self._cells
        v = pd.concat([x.cells for x in self._samples])
        self._cells = v
        return self._cells

class Sample(GenericSamples):
    def __init__(self,path,mydir,verbose=False):
        GenericSamples.__init__(self)
        self._path = path
        self._cells = None
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
            f = Frame(path,mydir,sample,frame,data,score,summary)
            self._frames[frame] = f
        if len(snames) > 1:
        	raise ValueError('Error multiple samples in folder '+path)
        self._sample_name = list(snames)[0]        
    @property
    def cells(self):
        if self._cells is not None: return self._cells
        v =  pd.concat([x.cells for x in self._frames.values()])
        self._cells = v
        return(v)

class Frame(GenericSamples):
    def __init__(self,path,mydir,sample,frame,seg_file,score_file,summary_file):
        #self._seg = pd.read_csv(seg_file,sep=",")
        GenericSamples.__init__(self)
        self._frame = frame
        self._sample = sample
        self._areas = None #cache
        self._phenotypes_present = None # cache for the value
        self._tissues_present = None # cache for the value
        self._seg = pd.read_csv(seg_file,"\t")
        #self._score = OrderedDict(pd.read_csv(score_file,"\t").iloc[0].to_dict())
        self._scores = OrderedDict()
        sfile = pd.read_csv(score_file,"\t")
        head = sfile.columns
        for row in sfile.itertuples(index=False):
            #row = row.to_dict()
            s = pd.Series(row,index=head)
            if s['Tissue Category'] in self._scores: raise ValueError('Same tissue is scored multiple times '+"\n"+str(s))
            self._scores[s['Tissue Category']] = OrderedDict(s.to_dict())
        if pd.read_csv(score_file,"\t").shape[0] > 1:
            raise ValueError("You need to fix code to allow for more than one thresolding in a single score file")
        self._summary = None
        # get the enumeration of the components from a pattern match
        # get the stains and thresholds
        self._stains = OrderedDict()
        for tissue in self._scores:
            if tissue not in self._stains:
                self._stains[tissue] = OrderedDict()
            checks = [re.match('(\S+)\s+',x).group(1) for x in list(self._scores[tissue].keys()) if re.match('\S+ Cell Compartment$',x)]
            for check in checks:
                compartment = check+' Cell Compartment'
                stain = check+' Stain Component'
                if compartment in self._scores[tissue]:
                    self._stains[tissue][self._scores[tissue][stain]] = OrderedDict({'compartment':self._scores[tissue][compartment],
                   	    'threshold':self._get_threshold(tissue,self._scores[tissue][stain])})#self._scores[tissue][stain]})
        # get the components
        self._components = []
        for name in self._seg.columns:
        	m = re.match('Entire Cell (.*) Mean \(Normalized Counts, Total Weighting\)',name)
        	if not m: continue
        	component = m.group(1)
        	if component not in self._components: self._components.append(component)
        # In the circumstance that the summary file exsts extract information
        if summary_file:
            self._summary = pd.read_csv(summary_file,sep="\t")

        ##### FINISHED READING IN THINGS NOW OUTPUT THINGS ##########
        keepers = ['Cell ID','Phenotype',
            'Cell X Position',
            'Cell Y Position',
            'Entire Cell Area (pixels)','Tissue Category']
        keepers2 = [x for x in self._seg.columns if re.search('Entire Cell.*Mean \(Normalized Counts, Total Weighting\)$',x)]
        keepers3 = [x for x in self._seg.columns if re.search('Mean \(Normalized Counts, Total Weighting\)$',x) and x not in keepers2]
        entire = OrderedDict()
        for cname in keepers2:
            m = re.match('Entire Cell\s+(.*) Mean \(Normalized Counts, Total Weighting\)$',cname)
            stain = m.group(1)
            v = self._seg[['Cell ID',cname]]
            v.columns = ['Cell ID','value']
            v = v.copy()
            for row in v.itertuples(index=False):
                if row[0] not in entire: entire[row[0]] = OrderedDict()
                entire[row[0]][stain]=row[1]
        compartments = OrderedDict()
        for cname in keepers3:
            if re.match('Entire Cell',cname): continue
            m = re.match('(\S+)\s+(.*) Mean \(Normalized Counts, Total Weighting\)$',cname)
            compartment = m.group(1)
            stain = m.group(2)
            v = self._seg[['Cell ID',cname]]
            v.columns = ['Cell ID','value']
            v = v.copy()
            for row in v.itertuples(index=False):
                if row[0] not in compartments: compartments[row[0]] = OrderedDict()
                if stain not in compartments[row[0]]: compartments[row[0]][stain] = OrderedDict()
                compartments[row[0]][stain][compartment] = row[1]
        v = self._seg[keepers].copy()
        v['compartment_values'] = v.apply(lambda x: compartments[x['Cell ID']],1)
        v['entire_cell_values'] = v.apply(lambda x: entire[x['Cell ID']],1)
        #v = self._seg[keepers+keepers2]
        v = v.rename(columns = {'Cell ID':'id',
            'Entire Cell Area (pixels)':'cell_area',
            'Cell X Position':'x',
            'Cell Y Position':'y',
            'Phenotype':'phenotype',
            'Tissue Category':'tissue'})
        v['frame'] = self._frame
        v['sample'] = self._sample
        #v['threshold_marker'] = np.nan
        #v['threshold_call'] = np.nan
        #v['full_phenotype'] = v['phenotype']
        v['frame_stains'] = None
        v['frame_stains'] = v.apply(lambda x: json.dumps(self._stains),1) 


        if self._summary is not None:
            #### Read our areas from the summary #####
            
            myareas = OrderedDict(self.frame_areas)
            ### Now we have our areas read lets put that data into things
            tissues_present = [x for x in myareas.keys() if x != 'All']
            v['tissue_area'] = v.apply(lambda x: myareas[x['tissue']],1)
            v['total_area'] = v.apply(lambda x: myareas['All'],1)
            v['tissues_present'] = v.apply(lambda x: json.dumps(tissues_present),1)
            ### Lets the phenotypes that are here
            phenotypes_present = [x for x in sorted(self._summary['Phenotype'].unique().tolist()) if x != 'All']
            v['phenotypes_present'] = v.apply(lambda x: json.dumps(phenotypes_present),1)
        else:
            v['tissue_area'] = np.nan
            v['total_area'] = np.nan
            v['phenotypes_present'] = json.dumps([])
            v['tissues_present'] = json.dumps({})
        v['folder'] = mydir.lstrip('/').lstrip('\\')
        self._cells = v
    @property
    def cells (self):
        return self._cells
    @property
    def frame_areas (self):
        if self._summary is None: raise ValueError("You need summary files present to get areas")
        if self._areas is not None: return self._areas
        df = self._summary.copy()
        mega = df.apply(lambda x: np.nan if float(x['Cell Density (per megapixel)']) == 0 else float(x['Total Cells'])/float(x['Cell Density (per megapixel)']),1) # cell area in mega pixels
        df['Summary Area Megapixels'] = mega
        # Lets ignore the cell specific things here
        #return(df[df['Phenotype']=='All'])
        df = df[['Tissue Category','Phenotype','Summary Area Megapixels']].\
            rename(columns={'Tissue Category':'tissue',
                            'Phenotype':'phenotype',
                            'Summary Area Megapixels':'tissue_area'
                           })
        df = df.loc[df['phenotype']=='All',['tissue','tissue_area']].set_index('tissue')['tissue_area'].to_dict()
        self._areas = df
        return(self._areas)
    @property
    def frame_components(self): return self._components
    def _get_threshold(self,tissue,stain):
        v = [x for x in self._scores[tissue].keys() if re.search(' Threshold$',x)]
        for entry in v:
            name = re.match('(.*) Threshold$',entry).group(1)
            if name == stain:
                return self._scores[tissue][entry]
        raise ValueError('did not find tissue and stain '+str(tissue)+' '+str(stain)+' '+str(self._scores))
    @property
    def frame_scores(self):
        v = self._score.copy()
        v['sample'] = self._sample
        v['frame'] = self._frame
        return pd.DataFrame(pd.Series(v)).transpose()


