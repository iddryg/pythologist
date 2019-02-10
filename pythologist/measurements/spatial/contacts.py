import pandas as pd
import numpy as np
from itertools import chain
from pythologist.measurements import Measurement


def _find_one(d):
    for k in d.keys():
        if d[k] == 1: return k
    return np.nan

class Contacts(Measurement):
    @staticmethod
    def _preprocess_dataframe(cdf,*args,**kwargs):
        mergeon = ['project_id','sample_id','frame_id','cell_index']
        subset = cdf.loc[~cdf['neighbors'].isna()].copy().reset_index(drop=True)
        present = subset[mergeon].drop_duplicates()
        def _get_items(x):
            v = x['neighbors'].items()
            return ([x.name]*len(v),list(x['neighbors'].keys()),list(x['neighbors'].values()))
        data = subset.dropna().apply(_get_items,1) 
        myindex = list(chain(*[x[0] for x in data]))
        myneighbor = list(chain(*[x[1] for x in data]))
        myshared = list(chain(*[x[2] for x in data]))
        data = pd.DataFrame({'db_id':myindex,'neighbor_cell_index':myneighbor,'edge_shared_pixels':myshared})
        base = subset[['project_id','project_name',
                   'sample_id','sample_name',
                   'frame_id','frame_name',
                   'cell_index','region_label','regions',
                   ]].merge(data,left_index=True,right_on='db_id').\
                   drop(columns='db_id')
        temp = cdf[['frame_id','cell_index','edge_length','phenotype_calls']].copy()
        temp['phenotype_calls'] = temp['phenotype_calls'].apply(lambda x: _find_one(x))
        temp = temp.loc[~temp['phenotype_calls'].isna()].rename(columns={'phenotype_calls':'phenotype'})
        merged = temp.merge(base,on=['frame_id','cell_index'])
        temp2 = temp.copy().rename(columns={'phenotype':'neighbor_phenotype','edge_length':'neighbor_edge_length','cell_index':'neighbor_cell_index'})
        merged = merged.merge(temp2,on=['frame_id','neighbor_cell_index'])
        return merged
    def counts(self):
        mergeon=['project_id',
                             'project_name',
                             'sample_id',
                             'sample_name',
                             'frame_id',
                             'frame_name',
                             'region_label']
        mr = self.measured_regions.copy()
        mr['_key'] = 1
        cnts = self.groupby(mergeon+['phenotype','neighbor_phenotype']).\
            count()[['cell_index']].rename(columns={'cell_index':'count'})
        cnts = cnts.reset_index()
        pheno1 = pd.DataFrame({'phenotype':self.measured_phenotypes})
        pheno1['_key'] = 1
        pheno2 = pheno1.copy().rename(columns={'phenotype':'neighbor_phenotype'})
        blank = mr.merge(pheno1,on='_key').merge(pheno2,on='_key').drop(columns='_key')
        cnts = blank.merge(cnts,on=mergeon+['phenotype','neighbor_phenotype'],how='left').fillna(0)
        return cnts