from pythologist.selection import SubsetLogic as SL
import pandas as pd
import numpy as np
import math

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
        'label'
        ]).apply(lambda x: {
            'frame_count':len(x['frame_id']),
            'count':np.sum(x['count']),
            'mean_density_mm2':np.mean(x['density_mm2']),
            'stderr_mm2':np.nan if len(x['frame_id'])==0 else np.std(x['density_mm2'])/math.sqrt(len(x['frame_id']))
        }).apply(pd.Series,1).reset_index()
    return output