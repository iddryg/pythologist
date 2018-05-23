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

def read_inForm(path,mpp=0.496,verbose=False,limit=None,sample_index=1):
    return InFormCellFrame.read_inForm(path,mpp=mpp,verbose=verbose,limit=limit,sample_index=sample_index)

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
def _modify_phenotypes(phenotypes_present,target,abbrev):
    keep = []
    for phenotype in json.loads(phenotypes_present):
        if phenotype == target:
            keep.append(phenotype+' '+abbrev+'+')
            keep.append(phenotype+' '+abbrev+'-')
        else: keep.append(phenotype)
    return json.dumps(keep)

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
    def read_inForm(path,mpp=0.496,verbose=False,limit=None,sample_index=1):
        """ path is the location of the folds
            """ 
        return InFormCellFrame(pythologist.read.SampleSet(path,verbose,limit,sample_index).cells,mpp=mpp)
    def write_inForm(self,path,overwrite=False):
        """ path is the location of the folds
            """ 
        return pythologist.write.write_inForm(self,path,overwrite=overwrite)
    #### Properties of the InFormCellFrame
    @property
    def samples(self):
        return None
    @property
    def frame_data(self):
        frame_general = self.df[['folder','sample','frame','total_area','tissues_present','phenotypes_present','frame_stains']].drop_duplicates(subset=['folder','sample','frame','tissue'])
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
        for row in self.df[['folder','sample','frame','tissue','tissues_present','frame_stains']].drop_duplicates(subset=['folder','sample','frame','tissue']).itertuples(index=False):
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
    def scored_stains(self):
        stains = set()
        sdict = self.frame_stains_to_dict()
        for stain_str in sdict:
            for tissue in sdict[stain_str]:
                for stain in sdict[stain_str][tissue]:
                    stains.add(stain)
        return sorted(list(stains))
    @property
    def all_stains(self):
        stains = set()
        for ecv in self['entire_cell_values']:
            stains |= set(list(ecv.keys()))
        return list(stains)
    @property
    def compartments(self):
        compartments = set()
        for com in self['compartment_areas']: compartments |= set(list(com.keys()))
        return list(compartments)
    @property
    def phenotypes(self):
        phenotypes = set()
        for pp in self['phenotypes_present'].unique():
            for phenotype in json.loads(pp): phenotypes.add(phenotype)
        return list(phenotypes)
    @property
    def tissues(self):
        tissues = set()
        for tissue_dict in [json.loads(x) for x in self['tissues_present'][self['tissues_present'].notnull()].unique()]:
            for t in tissue_dict.keys(): tissues.add(t)
        return list(tissues)
    @property
    def mpp(self): return self._mpp

    @property
    def df(self):
        return pd.DataFrame(self).copy()

    ### Drop a stain
    def drop_stain(self,stain):
        fsd = self.frame_stains_to_dict()
        rows = []
        for row in self.itertuples(index=False):
            s = pd.Series(row,index=self.columns)
            fs = fsd[s['frame_stains']]
            for tissue in fs.keys():
                if stain in fs[tissue]: del fs[tissue][stain]
            s['frame_stains'] = json.dumps(fs)
            ecv = s['entire_cell_values']
            if stain in ecv: del ecv[stain]
            s['entire_cell_values'] = ecv
            cv = s['compartment_values']
            if stain in cv: del cv[stain]
            s['compartment_values'] = cv
            rows.append(s)
        return InFormCellFrame(pd.DataFrame(rows),mpp=self.mpp)
    # For stains and tissues give all the cells values
    def zero_fill_values(self):
        rows = []
        all_stains = self.all_stains
        compartments = self.compartments
        for row in self.itertuples(index=False):
            s = pd.Series(row,index=self.columns)
            for stain in all_stains:
                if stain not in s['compartment_values']:
                    s['compartment_values'][stain] = OrderedDict()
                if stain not in s['entire_cell_values']:
                    s['entire_cell_values'][stain] = OrderedDict({'Min':0,'Mean':0,'Max':0,'Std Dev':0,'Total':0})
                for compartment in compartments:
                    if compartment not in s['compartment_values'][stain]: 
                        s['compartment_values'][stain][compartment] = OrderedDict({'Min':0,'Mean':0,'Max':0,'Std Dev':0,'Total':0})
            rows.append(s)
        return InFormCellFrame(pd.DataFrame(rows),mpp=self.mpp)




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
        #print(self['tissues_present'])
        #print(adict)
        anew = OrderedDict()
        for astr in adict:
            if old_name not in adict[astr]:
                anew[astr] = astr
                continue
            if new_name in adict[astr]: raise ValueError("error can't rename a tissue to one thats already there")
            v = OrderedDict()
            for tissue in adict[astr]:
                if tissue != old_name:
                    v[tissue] = adict[astr][tissue]
                    continue
                else:
                    v[new_name] = adict[astr][tissue]
            anew[astr] = json.dumps(v)
        #print(anew)
        df['tissues_present'] = df.apply(lambda x: anew[x['tissues_present']],1)
        #print(df['tissues_present'])
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
        mf2['phenotypes_present'] = mf2.apply(lambda x: _modify_phenotypes(x['phenotypes_present'],phenotype,abbrev),1)
        return InFormCellFrame(mf2.drop(columns=['new_phenotype']),mpp=self.mpp)

    def kNearestNeighborsCross(cf,phenotypes,k=1,threads=1,dlim=None):
        nn = pythologist.spatial.kNearestNeighborsCross(cf,phenotypes,k=k,threads=threads,dlim=dlim)
        return pythologist.spatial.CellFrameNearestNeighbors(cf,nn)
    def set_mpp(self,value):
        self._mpp = value
    @property
    def frame_counts(self):
        data = self.df
        data['tissues_present'] = data['tissues_present'].astype(str)
        data['frame_stains'] = data['frame_stains'].astype(str)
        data['phenotypes_present'] = data['phenotypes_present'].astype(str)

        # Assuming all phenotypes and all tissues could be present in all frames
        basic = data.groupby(['folder','sample','frame','tissue','phenotype']).first().reset_index()[['folder','sample','frame','tissue','phenotype','tissues_present']]

        cnts = data.groupby(['folder','sample','frame','tissue','phenotype']).count().\
             reset_index()[['folder','sample','frame','tissue','phenotype','id']].\
             rename(columns ={'id':'count'})
        cnts = cnts.merge(basic,on=['folder','sample','frame','tissue','phenotype'])
        cnts['tissue_area'] = cnts.apply(lambda x: json.loads(x['tissues_present'])[x['tissue']],1)
        # For each frame
        df = pd.DataFrame(self)[['folder','sample','frame','phenotypes_present','tissues_present']].groupby(['folder','sample','frame']).first().reset_index()
        empty = []
        for frame in df.itertuples(index=False):
            s = pd.Series(frame,df.columns)
            f = self[(self['folder']==s['folder'])&(self['sample']==s['sample'])&(self['frame']==s['frame'])]
            f = f.copy()
            tissues = list(json.loads(s["tissues_present"]).keys())
            td = json.loads(s["tissues_present"])
            phenotypes = json.loads(s["phenotypes_present"])
            folder = s['folder']
            sname = s["sample"]
            fname = s["frame"]
            for tissue in tissues:
                for phenotype in phenotypes:            
                    sub = cnts[(cnts['folder']==folder)&(cnts['sample']==sname)&(cnts['frame']==fname)&(cnts['tissue']==tissue)&(cnts['phenotype']==phenotype)]
                    subcnt = sub.shape[0]
                    if subcnt != 0: continue
                    g = pd.Series({'folder':folder,
                                   'sample':sname,
                                   'frame':fname,
                                   'tissue':tissue,
                                   'tissue_area':td[tissue],
                                   #'total_area':s['total_area'],
                                   'phenotype':phenotype,
                                   'count':0})
                    empty.append(g)

        out = pd.concat([cnts,pd.DataFrame(empty)],sort=True)\
              [['folder','sample','frame','tissue','tissue_area','phenotype','count']].\
            sort_values(['sample','frame','tissue','phenotype'])
        out['tissue_area'] = out['tissue_area'].fillna(0)
        out['tissue_density'] = out.apply(lambda x: np.nan if float(x['tissue_area']/1000000) == 0 else float(x['count'])/(float(x['tissue_area'])/1000000),1)
        out['tissue_area_mm2'] = out.apply(lambda x: (x['tissue_area']/1000000)/(self.mpp*self.mpp),1)
        out['tissue_density_mm2'] = out.apply(lambda x: x['tissue_density']/(self.mpp*self.mpp),1)
        #out['total_area'] = out['total_area'].fillna(0)
        return(out)
    @property
    def sample_counts(self):
        fc = self.frame_counts

        v = fc[fc['tissue_density'].notnull()].groupby(['folder','sample','tissue','phenotype']).\
            count().reset_index()[['folder','sample','tissue','phenotype','tissue_density']].\
            rename(columns={'tissue_density':'present_count'})
        basic = fc.groupby(['folder','sample','tissue','phenotype']).first().reset_index()[['folder','sample','tissue','phenotype']]
        mean = fc.groupby(['folder','sample','tissue','phenotype']).mean().reset_index()[['folder','sample','tissue','phenotype','tissue_density']].rename(columns={'tissue_density':'mean'})
        std = fc.groupby(['folder','sample','tissue','phenotype']).std().reset_index()[['folder','sample','tissue','phenotype','tissue_density']].rename(columns={'tissue_density':'std_dev'})
        cnt =  fc.groupby(['folder','sample','tissue','phenotype']).count().reset_index()[['folder','sample','tissue','phenotype','count']].rename(columns={'count':'frame_count'})
        out = basic.merge(mean,on=['folder','sample','tissue','phenotype']).merge(std,on=['folder','sample','tissue','phenotype']).merge(cnt,on=['folder','sample','tissue','phenotype'])
        out = out.merge(v,on=['folder','sample','tissue','phenotype'],how='left')
        out['present_count'] = out['present_count'].fillna(0).astype(int)
        out['std_err'] = out.apply(lambda x: np.nan if x['present_count'] == 0 else x['std_dev']/(math.sqrt(x['present_count'])),1)
        out['mean_mm2'] = out.apply(lambda x: x['mean']/(self.mpp*self.mpp),1)
        out['std_dev_mm2'] = out.apply(lambda x: x['std_dev']/(self.mpp*self.mpp),1)
        out['std_err_mm2'] = out.apply(lambda x: x['std_err']/(self.mpp*self.mpp),1)
        return out
    def merge_phenotype_data(self,replacement_idf,phenotype_old,phenotype_replacement,scale=1):
        #Assumes sample and frame names are unique and there are not multiple folders
        # Be careful the phenotype you add in will not be properly thresholded, and if a stain was not measured if you do not specifically remove it, it will be set to zero
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
        ### We need to set the frame stains in keepers 1 and keepers 2 to what they are in self's frames
        both = pd.concat([keepers1,keepers2],sort=True)
        rows = []
        for row in both.itertuples(index=False):
            s = pd.Series(row,index=both.columns)
            frame = self[(self['sample']==s['sample'])&(self['frame']==s['frame'])&(self['tissue']==s['tissue'])]
            if frame.shape[0] == 0: s['frame_stains'] = json.dumps({s['tissue']:OrderedDict()})
            else: s['frame_stains'] = frame.iloc[0]['frame_stains']
            rows.append(s)
        both = pd.DataFrame(rows)

        
        keepers_alt = current[~current['index'].isin(drop1['index'])][['sample','frame','id']]
        mdf = pd.concat([both,
                        self.df.merge(keepers_alt,on=['sample','frame','id'])],sort=True)
        phenotypes = json.dumps(list(mdf['phenotype'].dropna().unique()))
        mdf['phenotypes_present'] = phenotypes
        mdf['folder'] = mdf.apply(lambda x: x['sample'],1)
        organized = [] 
        for sample in mdf['sample'].unique():
            for frame in mdf.loc[mdf['sample']==sample,'frame'].unique():
                frame = mdf[(mdf['sample']==sample)&(mdf['frame']==frame)].copy().sort_values(['x','y']).reset_index(drop=True)
                frame['id'] = range(1,frame.shape[0]+1,1)
                organized.append(frame)
        return InFormCellFrame(pd.concat(organized,sort=True),mpp=self.mpp)