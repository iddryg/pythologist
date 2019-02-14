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
                   'cell_index','region_label',
                   ]].merge(data,left_index=True,right_on='db_id').\
                   drop(columns='db_id')
        temp = cdf[['frame_id','cell_index','edge_length','phenotype_calls']].copy()
        temp['phenotype_calls'] = temp['phenotype_calls'].apply(lambda x: _find_one(x))
        temp = temp.loc[~temp['phenotype_calls'].isna()].rename(columns={'phenotype_calls':'phenotype_label'})
        merged = temp.merge(base,on=['frame_id','cell_index'])
        temp2 = temp.copy().rename(columns={'phenotype_label':'neighbor_phenotype_label',
                                            'edge_length':'neighbor_edge_length',
                                            'cell_index':'neighbor_cell_index'})
        merged = merged.merge(temp2,on=['frame_id','neighbor_cell_index'])
        return merged
    def _proportions(self,mergeon):
        tot = self.groupby(mergeon+['phenotype_label']).\
            count()[['cell_index']].rename(columns={'cell_index':'total'})
        mr = self.measured_regions[mergeon].drop_duplicates()
        mr['_key'] = 1
        mp = pd.DataFrame({'phenotype_label':self.measured_phenotypes})
        mp['_key'] = 1
        mn = pd.DataFrame({'neighbor_phenotype':self.measured_phenotypes})
        mn['_key'] = 1
        tot = mr.merge(mp,on='_key').\
            merge(tot,on=mergeon+['phenotype_label'],how='left').fillna(0).drop(columns='_key')
        cnt = self.groupby(mergeon+['phenotype_label','neighbor_phenotype_label']).\
            count()[['cell_index']].rename(columns={'cell_index':'count'})
        cnt = mr.merge(mp,on='_key').merge(mn,on='_key').\
            merge(cnt,on=mergeon+['phenotype_label','neighbor_phenotype_label'],how='left').fillna(0)
        cnt = cnt.merge(tot,on=mergeon+['phenotype_label']).drop(columns='_key')
        cnt['fraction'] = cnt.apply(lambda x: 
                np.nan if x['total'] == 0 else x['count']/x['total']
            ,1)
        return cnt
    def frame_proportions(self):
        return self._proportions(self.cdf.frame_columns+['region_label'])
    def sample_proportions(self):
        return self._proportions(self.cdf.sample_columns+['region_label'])
    def project_proportions(self):
        return self._proportions(self.cdf.project_columns+['region_label'])

    def frame_counts(self):
        mergeon = self.cdf.frame_columns+['region_label']
        mr = self.measured_regions.copy()
        mr['_key'] = 1
        cnts = self.groupby(mergeon+['phenotype_label','neighbor_phenotype_label']).\
            count()[['cell_index']].rename(columns={'cell_index':'count'})
        cnts = cnts.reset_index()
        pheno1 = pd.DataFrame({'phenotype_label':self.measured_phenotypes})
        pheno1['_key'] = 1
        pheno2 = pheno1.copy().rename(columns={'phenotype_label':'neighbor_phenotype_label'})
        blank = mr.merge(pheno1,on='_key').merge(pheno2,on='_key').drop(columns='_key')
        cnts = blank.merge(cnts,on=mergeon+['phenotype_label','neighbor_phenotype_label'],how='left').fillna(0)
        cnts['region_area_mm2'] = cnts.apply(lambda x: 
            (x['region_area_pixels']/1000000)*(self.microns_per_pixel*self.microns_per_pixel),1)
        cnts['density_mm2'] = cnts.apply(lambda x: x['count']/x['region_area_mm2'],1)
        return cnts
    def sample_counts(self):
        mergeon = self.cdf.sample_columns+['region_label','phenotype_label','neighbor_phenotype_label']
        fc = self.measured_regions[self.cdf.frame_columns].\
            drop_duplicates().groupby(self.cdf.sample_columns).\
            count()[['frame_id']].rename(columns={'frame_id':'frame_count'})
        cnts = self.frame_counts().groupby(mergeon).\
            apply(lambda x:
                pd.Series(dict(zip(
                    ['cummulative_count',
                     'cummulative_region_area_pixels',
                     'cummulative_region_area_mm2',
                     'cummulative_density_mm2',
                     'mean_density_mm2',
                     'stddev_density_mm2',
                     'stderr_density_mm2',
                     'measured_frame_count'
                    ],
                    [
                     x['count'].sum(),
                     x['region_area_pixels'].sum(),
                     x['region_area_mm2'].sum(),
                     x['count'].sum()/x['region_area_mm2'].sum(),
                     x['density_mm2'].mean(),
                     x['density_mm2'].std(),
                     x['density_mm2'].std()/np.sqrt(len(x['density_mm2'])),
                     len(x['density_mm2'])
                    ]
                )))
            ).reset_index()
        cnts = cnts.merge(fc,on=self.cdf.sample_columns)
        return cnts
    def threshold(self,phenotype,contact_label=None):
        if contact_label is None: contact_label = phenotype+'/contact'
        def _add_score(d,value,label):
            d[label] = int(value)
            return d
        # for the given phenotype, define whether a cell is touching or not 
        cdf = self.cdf.copy()
        mergeon = cdf.frame_columns+['cell_index']
        contacts = self.loc[self['neighbor_phenotype_label']==phenotype,mergeon].drop_duplicates()
        contacts['_threshold'] = 1
        cdf = cdf.merge(contacts,on=mergeon,how='left')
        cdf.loc[cdf['_threshold'].isna(),'_threshold'] = 0
        cdf['scored_calls'] = cdf.apply(lambda x:
            _add_score(x['scored_calls'],x['_threshold'],contact_label)
        ,1)
        cdf.microns_per_pixel = self.microns_per_pixel
        return cdf.drop(columns='_threshold')
