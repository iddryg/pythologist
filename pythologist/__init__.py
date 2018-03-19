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
import pythologist.read
import pythologist.write

def read_inForm(path,mpp=0.496,verbose=False,limit=None,type="Vectra"):
    return InFormCellFrame.read_inForm(path,mpp=mpp,verbose=verbose,limit=limit,type=type)

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
    _metadata = ['_mpp']
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
        return InFormCellFrame(pd.DataFrame(self).copy(),mpp=self.mpp)
    def to_hdf(self,path):
        string_version = self.copy()
        string_version['compartment_values'] = string_version.apply(lambda x: 
            json.dumps(x['compartment_values']),1)
        string_version['entire_cell_values'] = string_version.apply(lambda x: 
            json.dumps(x['entire_cell_values']),1)
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
    def read_inForm(path,mpp=0.496,verbose=False,limit=None,type="Vectra"):
        """ path is the location of the folds
            """ 
        return InFormCellFrame(pythologist.read.SampleSet(path,verbose,limit,type=type).cells,mpp=mpp)
    def write_inForm(self,path,type="Vectra",overwrite=False):
        """ path is the location of the folds
            """ 
        return pythologist.write.write_inForm(self,path,type=type,overwrite=overwrite)
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
    @property
    def mpp(self): return self._mpp

    @property
    def df(self):
        return pd.DataFrame(self).copy()

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
        mf.loc[mf.apply(lambda x: x['compartment_values'][stain][x['compartment']]['Mean'] > x['threshold'],1),'gate'] = '+'
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
    def merge_phenotype_data(self,replacement_idf,phenotype_old,phenotype_replacement,scale=1):
        #Assumes sample and frame names are unique and there are not multiple folders
        # for each
        rdf = replacement_idf[(replacement_idf['phenotype']==phenotype_replacement)]
        replace = rdf.df[['sample','frame','x','y','id','cell_area']].drop_duplicates()
        replace['cell_radius'] = replace.apply(lambda x: math.sqrt(x['cell_area']/math.pi),1)
        
        current = self.df
        current['cell_radius'] = current.apply(lambda x: math.sqrt(x['cell_area']/math.pi),1)
        current = current.reset_index(drop=True).reset_index()
        can = current.loc[current['phenotype']==phenotype_old,['sample','frame','x','y','id','cell_radius','index']]
        alone = []
        match = []
        dropped = set()
        for row in replace.itertuples(index=False):
            s = pd.Series(row,index=replace.columns)
            found = can[(can['sample']==s['sample'])&
                (can['frame']==s['frame'])&
                (can['x']>s['x']-(s['cell_radius']+can['cell_radius'])*scale)&
                (can['x']<s['x']+(s['cell_radius']+can['cell_radius'])*scale)&
                (can['y']>s['y']-(s['cell_radius']+can['cell_radius'])*scale)&
                (can['y']<s['y']+(s['cell_radius']+can['cell_radius'])*scale)&
                (~can['id'].isin(list(dropped)))]
            if found.shape[0] == 0:
                alone.append(s)
                continue
            found = found.copy()
            found['x1'] = s['x']
            found['y1'] = s['y']
            found['id1'] = s['id']
            found['distance'] = found.apply(lambda x: 
                math.sqrt(((x['x1']-x['x'])*(x['x1']-x['x']))+((x['y1']-x['y'])*(x['y1']-x['y']))),1)
            found=found.sort_values('distance').iloc[0]
            dropped.add(found['id']) # These IDs have already been flagged to be dropped
            match.append(found)
        keepers1 = replacement_idf.df.merge(pd.DataFrame(match).drop(columns=['id']).\
            rename(columns={'id1':'id'})[['sample','frame','id']],on=['sample','frame','id'])
        drop1 = current.merge(pd.DataFrame(match),on=['sample','frame','id','index'])[['sample','frame','id','index']]
        keepers2 = replacement_idf.df.merge(pd.DataFrame(alone)[['sample','frame','id']],on=['sample','frame','id'])
        keepers_alt = current[~current['index'].isin(drop1['index'])][['sample','frame','id']]
        mdf = pd.concat([keepers1,
                        keepers2,
                        self.df.merge(keepers_alt,on=['sample','frame','id'])])
        phenotypes = json.dumps(list(mdf['phenotype'].dropna().unique()))
        mdf['phenotypes_present'] = phenotypes
        mdf['folder'] = mdf.apply(lambda x: x['sample'],1)
        organized = [] 
        for sample in mdf['sample'].unique():
            for frame in mdf.loc[mdf['sample']==sample,'frame'].unique():
                frame = mdf[(mdf['sample']==sample)&(mdf['frame']==frame)].copy().sort_values(['x','y']).reset_index(drop=True)
                frame['id'] = range(1,frame.shape[0]+1,1)
                organized.append(frame)
        return InFormCellFrame(pd.concat(organized),mpp=self.mpp)