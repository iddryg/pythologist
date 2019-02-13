import pandas as pd
import numpy as np
import sys, json, h5py
from pythologist.selection import SubsetLogic
from pythologist.measurements.counts import Counts
from pythologist.measurements.spatial.contacts import Contacts
from pythologist.measurements.spatial.nearestneighbors import NearestNeighbors

class CellDataFrame(pd.DataFrame):
    _metadata = ['_microns_per_pixel'] # for extending dataframe to include this property
    @property
    def _constructor(self):
        return CellDataFrame
    def __init__(self,*args,**kw):
        kwcopy = kw.copy()
        super(CellDataFrame,self).__init__(*args,**kwcopy)

    @property
    def frame_columns(self):
        # These fields isolate individual images
        return ['project_id','project_name',
                'sample_id','sample_name',
                'frame_id','frame_name']
    @property
    def sample_columns(self):
        # These fields isolate individual samples
        return ['project_id','project_name',
                'sample_id','sample_name']
    @property
    def project_columns(self):
        # These fields isolate projects
        return ['project_id','project_name']

    def to_hdf(self,path,key,mode='a'):
        # overwrite pandas to write to a dataframe
        pd.DataFrame(self.serialize()).to_hdf(path,key,mode=mode,format='table',complib='zlib',complevel=9)
        f = h5py.File(path,'r+')
        f[key].attrs["microns_per_pixel"] = float(self.microns_per_pixel) if self.microns_per_pixel is not None else np.nan
        f.close()

    @classmethod
    def read_hdf(cls,path,key=None):
        df = pd.read_hdf(path,key)
        df['scored_calls'] = df['scored_calls'].apply(lambda x: json.loads(x))
        df['channel_values'] = df['channel_values'].apply(lambda x: json.loads(x))
        df['regions'] = df['regions'].apply(lambda x: json.loads(x))
        df['phenotype_calls'] = df['phenotype_calls'].apply(lambda x: json.loads(x))
        df['neighbors'] = df['neighbors'].apply(lambda x: json.loads(x))
        df['neighbors'] = df['neighbors'].apply(lambda x:
                np.nan if not isinstance(x,dict) else dict(zip([int(y) for y in x.keys()],x.values()))
            )
        df = cls(df)
        f = h5py.File(path,'r')
        mpp = f[key].attrs["microns_per_pixel"]
        if not np.isnan(mpp): df.microns_per_pixel = mpp
        f.close()
        return df

    def serialize(self):
        df = self.copy()
        df['scored_calls'] = df['scored_calls'].apply(lambda x: json.dumps(x))
        df['channel_values'] = df['channel_values'].apply(lambda x: json.dumps(x))
        df['regions'] = df['regions'].apply(lambda x: json.dumps(x))
        df['phenotype_calls'] = df['phenotype_calls'].apply(lambda x: json.dumps(x))
        df['neighbors'] = df['neighbors'].apply(lambda x: json.dumps(x))
        return df

    @property
    def microns_per_pixel(self):
        if not hasattr(self,'_microns_per_pixel'): return None
        return self._microns_per_pixel
    @microns_per_pixel.setter
    def microns_per_pixel(self,value):
        self._microns_per_pixel = value

    def is_uniform(self,verbose=True):
        uni = pd.Series(self['phenotype_calls'].apply(lambda x: json.dumps(x)).unique()).\
            apply(lambda x: json.loads(x)).apply(lambda x: tuple(sorted(x.keys()))).unique()
        if len(uni) > 1: 
            if verbose: sys.stderr.write("WARNING: phenotypes differ across the dataframe \n"+str(uni)+"\n")
            return False
        uni = pd.Series(self['scored_calls'].apply(lambda x: json.dumps(x)).unique()).\
            apply(lambda x: json.loads(x)).apply(lambda x: tuple(sorted(x.keys()))).unique()
        if len(uni) > 1: 
            if verbose: sys.stderr.write("WARNING: scored_calls differ across the dataframe \n"+str(uni)+"\n")
            return False
        return True

    @property
    def phenotypes(self):
        # The mutually exclusive phenotypes present in the CellDataFrame
        return _extract_unique_keys_from_series(self['phenotype_calls'])

    @property
    def scored_names(self):
        return _extract_unique_keys_from_series(self['scored_calls'])

    @property
    def regions(self):
        return _extract_unique_keys_from_series(self['regions'])

    def get_measured_regions(self):
        # get measurable areas
        mergeon = ['project_id','project_name',
                'sample_id','sample_name',
                'frame_id','frame_name',
                ]
        temp = self.loc[:,mergeon+['regions']].\
            set_index(mergeon)['regions'].apply(json.dumps).\
            reset_index().drop_duplicates()
        temp['regions'] = temp['regions'].apply(json.loads)
        rows = []
        for i,r in temp.iterrows():
            for label in r['regions']:
                a = list(r.index)
                b = list(r.values)
                a = a+['region_label','region_area_pixels']
                b = b+[label,r['regions'][label]]
                rows.append(dict(zip(a,b)))
        rows = pd.DataFrame(rows).drop(columns='regions').\
            drop_duplicates()[mergeon+['region_label','region_area_pixels']]
        rows = rows.loc[rows['region_area_pixels']>0].copy()
        return rows

    def nearestneighbors(self,*args,**kwargs):
        n = NearestNeighbors.read_cellframe(self,*args,**kwargs)
        if 'measured_regions' in kwargs: n.measured_regions = kwargs['measured_regions']
        else: n.measured_regions = self.get_measured_regions()
        if 'measured_phenotypes' in kwargs: n.measured_phenotypes = kwargs['measured_phenotypes']
        else: n.measured_phenotypes = self.phenotypes
        n.microns_per_pixel = self.microns_per_pixel
        return n

    def contacts(self,*args,**kwargs):
        n = Contacts.read_cellframe(self)
        if 'measured_regions' in kwargs: n.measured_regions = kwargs['measured_regions']
        else: n.measured_regions = self.get_measured_regions()
        if 'measured_phenotypes' in kwargs: n.measured_phenotypes = kwargs['measured_phenotypes']
        else: n.measured_phenotypes = self.phenotypes
        n.microns_per_pixel = self.microns_per_pixel
        return n

    def counts(self,*args,**kwargs):
        n = Counts.read_cellframe(self)
        if 'measured_regions' in kwargs: n.measured_regions = kwargs['measured_regions']
        else: n.measured_regions = self.get_measured_regions()
        if 'measured_phenotypes' in kwargs: n.measured_phenotypes = kwargs['measured_phenotypes']
        else: n.measured_phenotypes = self.phenotypes
        n.microns_per_pixel = self.microns_per_pixel
        return n

    def _shuffle_ids(self):
        together = []
        for frame_id in self['frame_id'].unique():
            v1 = self.loc[self['frame_id']==frame_id,['cell_index']].copy().reset_index(drop=True)
            v2 = v1.copy().sample(frac=1)
            v1['next_index'] = list(v2.index)
            v1['frame_id'] = frame_id
            together.append(v1)
        return pd.concat(together)

    ### Modifying functions
    def merge_scores(self,df_addition,reference_markers='all',
                                      addition_markers='all',on=['project_name','sample_name','frame_name','cell_index']):
        if isinstance(reference_markers, str):
            reference_markers = self.scored_names
        elif reference_markers is None: reference_markers = []
        if isinstance(addition_markers, str):
            addition_markers = df_addition.scored_names
        elif additionmarkers is None: addition_markers = []

        df_addition = df_addition.copy()
        df_addition['_key'] = 1
        df = self.merge(df_addition[['scored_calls','_key']+on].rename(columns={'scored_calls':'_addition'}),
                            on = on,
                            how = 'left'
                        )
        df['_sub1'] = self['scored_calls'].apply(lambda x:
                dict((k,x[k]) for k in reference_markers)
            )
        df['_sub2'] = df_addition['scored_calls'].apply(lambda x:
                dict((k,x[k]) for k in addition_markers)
            )
        # combine the two dictionaries
        df['scored_calls'] = df.apply(lambda x:
                {**x['_sub1'],**x['_sub2']}                    
            ,1)
        df = df.drop(columns=['_sub1','_sub2','_addition'])

        df = df.drop(columns='_key').copy(),df[df['_key'].isna()].drop(columns='_key').copy()
        return df

    def rename_scored_calls(self,change):
        # input dictionary change with {<current name>:<new name>} format, new name must not already exist
        output = self.copy()
        output['scored_calls'] = output.apply(lambda x:
          _dict_rename(x['scored_calls'],change)
          ,1)
        return output

    def zero_fill_missing_phenotypes(self):
        # Fill in missing phenotypes and scored types by listing any missing data as negative
        if self.is_uniform(verbose=False): return self.copy()
        output = self.copy()
        def _do_fill(d,names):
            old_names = list(d.keys())
            old_values = list(d.values())
            missing = set(names)-set(old_names)
            return dict(zip(old_names+list(missing),old_values+([0]*len(missing))))
        ## Need to make these uniform
        pnames = self.phenotypes
        output['phenotype_calls']= output.apply(lambda x:
            _do_fill(x['phenotype_calls'],pnames)
            ,1)
        return output

    def subset(self,logic,update=False):
        # subset create a specific phenotype based on a logic
        # logic is a 'SubsetLogic' class
        # take union of all the phenotypes listed.  If none are listed use all phenotypes.
        # take the intersection of all the scored calls
        pnames = self.phenotypes
        snames = self.scored_names
        data = self.copy()
        values = []
        phenotypes = logic.phenotypes
        if len(phenotypes)==0: phenotypes = pnames
        removing = set(self.phenotypes)-set(phenotypes)
        for k in phenotypes:
            if k not in pnames: raise ValueError("phenotype must exist in defined")
            temp = data.loc[data['phenotype_calls'].apply(lambda x: x[k]==1)].copy()
            if len(removing) > 0:
                temp['phenotype_calls'] = temp.apply(lambda x:
                    dict([(k,v) for k,v in x['phenotype_calls'].items() if k not in removing])
                    ,1)
            values.append(temp)
        data = pd.concat(values)
        for k,v in logic.scored_calls.items():
            if k not in snames: raise ValueError("Scored name must exist in defined")
            filter = 0 if v == '-' else 1
            data = data.loc[data['scored_calls'].apply(lambda x: x[k]==filter)]
        data.microns_per_pixel = self.microns_per_pixel
        if update: data['phenotype_calls'] = data['phenotype_calls'].apply(lambda x: {logic.label:1})
        return data

    def threshold(self,phenotype,scored_name,positive_label=None,negative_label=None):
        # split a phenotype on a scored_call and if no label is specified
        #       use the format '<phenotype> <scored_call><+/->'
        # to specify a label give the positive and negative label
        if positive_label is None and negative_label is not None or \
           negative_label is None and positive_label is not None: raise ValueError("Error if you want to specify labels, give both positive and negative")
        if phenotype not in self.phenotypes: raise ValueError("Error phenotype "+str(phenotype)+" is not in the data.")
        if scored_name not in self.scored_names: raise ValueError("Error scored_name "+str(scored_name)+" is not in the data.")
        if positive_label is None and negative_label is None:
            positive_label = phenotype+' '+scored_name+'+'
            negative_label = phenotype+' '+scored_name+'-'
        elif positive_label == negative_label: raise ValueError("Cant have the same label for positive and negative.")
        def _swap_in(d,pheno,scored,phenotype_calls,scored_calls,pos,neg):
            if pheno not in phenotype_calls.keys(): return d
            keepers = [(k,v) for k,v in phenotype_calls.items() if k!=phenotype]
            if scored not in scored_calls.keys(): raise ValueError("Error scored calls are not unified across samples")
            scored_value = scored_calls[scored]
            phenotype_value = phenotype_calls[pheno]
            if phenotype_value == 0:
                keepers += [(pos,0),(neg,0)]
            elif scored_value == 1:
                keepers += [(pos,1),(neg,0)]
            elif scored_value == 0:
                keepers += [(pos,0),(neg,1)]
            else: raise ValueError("Format error.  These values should only ever be zero or one.")
            return dict(keepers)
        data = self.copy()
        data['phenotype_calls'] = self.apply(lambda x:
                _swap_in(x,phenotype,scored_name,x['phenotype_calls'],x['scored_calls'],positive_label,negative_label)
            ,1)
        def _set_label(d):
            vals = [k for k,v in d.items() if v==1]
            return np.nan if len(vals) == 0 else vals[0]
        data['phenotype_label'] = data.apply(lambda x:
                _set_label(x['phenotype_calls'])
            ,1)
        return data.copy()

    def collapse_phenotypes(self,input_phenotype_labels,output_phenotype_label,verbose=True):
        # Rename one or more input phenotypes to a single output phenotype
        if isinstance(input_phenotype_labels,str): input_phenotype_labels = [input_phenotype_labels]
        bad_phenotypes = set(input_phenotype_labels)-set(self.phenotypes)
        if len(bad_phenotypes) > 0: raise ValueError("Error phenotype(s) "+str(bad_phenotypes)+" are not in the data.")
        data = self.copy()
        if len(input_phenotype_labels) == 0: return data
        def _swap_in(d,inputs,output):
            # Get the keys we need to merge together
            overlap = set(d.keys()).intersection(inputs)
            # if there are none to merge we're done already
            if len(overlap) == 0: return d
            keepers = [(k,v) for k,v in d.items() if k not in inputs]
            # combine anything thats not a keeper
            return dict(keepers+\
                        [(output_phenotype_label,max([d[x] for x in overlap]))])
        data['phenotype_calls'] = data.apply(lambda x:
            _swap_in(x['phenotype_calls'],input_phenotype_labels,output_phenotype_label)
            ,1)
        return data
    def rename_phenotype(self,*args,**kwargs): 
        """simple alias for collapse phenotypes"""
        return self.collapse_phenotypes(*args,**kwargs)

    def combine_regions(self,input_region_labels,output_region_label,verbose=True):
        # Rename one or more input phenotypes to a single output phenotype
        if isinstance(input_region_labels,str): input_region_labels = [input_region_labels]
        bad_regions = set(input_region_labels)-set(self.regions)
        if len(bad_regions) > 0: raise ValueError("Error regions(s) "+str(bad_regions)+" are not in the data.")
        data = self.copy()
        if len(input_region_labels) == 0: return data
        def _swap_in(d,inputs,output):
            # Get the keys we need to merge together
            overlap = set(d.keys()).intersection(inputs)
            # if there are none to merge we're done already
            if len(overlap) == 0: return d
            keepers = [(k,v) for k,v in d.items() if k not in inputs]
            # combine anything thats not a keeper
            return dict(keepers+\
                        [(output_region_label,sum([d[x] for x in overlap]))])
        data['regions'] = data.apply(lambda x:
            _swap_in(x['regions'],input_region_labels,output_region_label)
            ,1)
        data.loc[data['region_label'].isin(input_region_labels),'region_label'] = output_region_label
        return data
    def rename_region(self,*args,**kwargs): 
        """simple alias for combine phenotypes"""
        return self.combine_regions(*args,**kwargs)

def _extract_unique_keys_from_series(s):
    uni = pd.Series(s.apply(lambda x: json.dumps(x)).unique()).\
            apply(lambda x: json.loads(x)).apply(lambda x: set(sorted(x.keys())))
    return sorted(list(set().union(*list(uni))))
def _dict_rename(old,change):
    new_keys = [x if x not in change else change[x] for x in old.keys()]
    return dict(zip(new_keys, old.values()))




