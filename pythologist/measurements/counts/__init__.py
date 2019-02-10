from pythologist.selection import SubsetLogic as SL
import pandas as pd
import numpy as np
import math
from pythologist.measurements import Measurement

class Counts(Measurement):
    @staticmethod
    def _preprocess_dataframe(cdf,*args,**kwargs):
        # set our phenotype labels
        data = pd.DataFrame(cdf) # we don't need to do anything special with the dataframe for counting
        data['phenotype_label'] = data.apply(lambda x: 
                [k for k,v in x['phenotype_calls'].items() if v==1]
            ,1).apply(lambda x: np.nan if len(x)==0 else x[0])
        return data
    def frame_counts(self,subsets=None):
        mergeon = ['project_id',
                   'project_name',
                   'sample_id',
                   'sample_name',
                   'frame_id',
                   'frame_name',
                   'region_label']
        if subsets is None:
            cnts = self.groupby(mergeon+['phenotype_label']).count()[['cell_index']].\
                rename(columns={'cell_index':'count'})
            mr = self.measured_regions
            mr['_key'] =  1
            mp = pd.DataFrame({'phenotype_label':self.measured_phenotypes})
            mp['_key'] = 1
            mr = mr.merge(mp,on='_key').drop(columns='_key')
            cnts = mr.merge(cnts,on=mergeon+['phenotype_label'],how='left').fillna(0)
        else:
             # Use subsets
            if isinstance(subsets,SL): subsets=[subsets]
            cnts = []
            labels = set([s.label for s in subsets])
            for x in subsets: 
                if x.label is None: raise ValueError("Subsets must be named")
            if len(labels) != len(subsets): raise ValueError("Subsets must be uniquely named.")
            for sl in subsets:
                df = self.cellframe.subset(sl)
                df = df.groupby(mergeon).count()[['cell_index']].\
                    rename(columns={'cell_index':'count'}).reset_index()
                df = self.measured_regions.merge(df,on=mergeon,how='left').fillna(0)
                df['phenotype_label'] = sl.label
                cnts.append(df)
            cnts = pd.concat(cnts)
        cnts = cnts[mergeon+['region_area_pixels','phenotype_label','count']]
        cnts['region_area_mm2'] = cnts.apply(lambda x: 
            (x['region_area_pixels']/1000000)/(self.microns_per_pixel*self.microns_per_pixel),1)
        cnts['density_mm2'] = cnts.apply(lambda x: x['count']/x['region_area_mm2'],1)
        return cnts


def frame_counts(cdf,subsets=None,ignore_empty_phenotypes=True):
    if ignore_empty_phenotypes:
        cdf = cdf.loc[cdf['phenotype_calls'].apply(lambda x: len(x.keys())>0 and max(x.values())>0)].copy()
    if cdf.microns_per_pixel is None: 
        raise ValueError("microns_per_pixel must be set to get counts")
    cdf['region_area'] = cdf.apply(lambda x: x['regions'][x['region_label']],1)
    # If no subsets are given return the phenotype counts
    if subsets is None: 
        subsets = []
        for phenotype in cdf.phenotypes:
            subsets.append(SL(label=phenotype,phenotypes={phenotype:'+'}))
    elif isinstance(subsets,SL): subsets=[subsets]
    mergeon = ['project_id',
               'project_name',
               'sample_id',
               'sample_name',
               'frame_id',
               'frame_name',
               'region_label',
               'region_area']
    frames_present = cdf[mergeon].drop_duplicates()

    counts = []
    for sl in subsets:
        df = cdf.subset(sl)
        df = pd.DataFrame(df)
        df = df.groupby(mergeon)[['cell_index']].count().\
             rename(columns={'cell_index':'count'}).reset_index()
        df = frames_present.merge(df,on=mergeon,how='left').fillna(0)
        df['label'] = sl.label
        counts.append(df)
    counts = pd.concat(counts)
    counts['microns_per_pixel'] = cdf.microns_per_pixel
    counts['region_area_mm2'] = counts.apply(lambda x: 
        (x['region_area']/1000000)/(cdf.microns_per_pixel*cdf.microns_per_pixel),1)
    counts['density_mm2'] = counts.apply(lambda x: x['count']/x['region_area_mm2'],1)
    return counts

def sample_counts(cdf,subsets=None,ignore_empty_phenotypes=True):
    fc = frame_counts(cdf,subsets=subsets,ignore_empty_phenotypes=ignore_empty_phenotypes).copy()
    output = fc.groupby([
        'project_id',
        'project_name',
        'sample_id',
        'sample_name',
        'region_label',
        'label'
        ]).apply(lambda x: {
            'frame_count':len(x['frame_id']),
            'total_count':np.sum(x['count']),
            'total_area_mm2':np.sum(x['region_area_mm2']),
            'total_density_mm2':np.nan if np.sum(x['region_area_mm2']) == 0 else np.sum(x['count'])/np.sum(x['region_area_mm2']),
            'mean_density_mm2':np.mean(x['density_mm2']),
            'std_err_mm2':np.nan if len(x['frame_id'])==0 else np.std(x['density_mm2'])/math.sqrt(len(x['frame_id']))
        }).apply(pd.Series,1).reset_index()
    return output