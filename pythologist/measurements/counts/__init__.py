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
        mergeon = self.cdf.frame_columns+['region_label']
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
            (x['region_area_pixels']/1000000)*(self.microns_per_pixel*self.microns_per_pixel),1)
        cnts['density_mm2'] = cnts.apply(lambda x: x['count']/x['region_area_mm2'],1)
        return cnts

    def sample_counts(self,subsets=None):
        mergeon = self.cdf.sample_columns+['region_label']
        fc = self.measured_regions[self.cdf.frame_columns+['region_label']].drop_duplicates().groupby(mergeon).\
            count()[['frame_id']].rename(columns={'frame_id':'frame_count'}).\
            reset_index()
        cnts = self.frame_counts(subsets=subsets).groupby(mergeon+['phenotype_label']).\
            apply(lambda x:
                pd.Series(dict(zip(
                    [
                     'cummulative_region_area_pixels',
                     'cummulative_region_area_mm2',
                     'cummulative_count',
                     'cummulative_density_mm2',
                     'mean_density_mm2',
                     'stddev_density_mm2',
                     'stderr_density_mm2',
                     'measured_frame_count'
                    ],
                    [
                     x['region_area_pixels'].sum(),
                     x['region_area_mm2'].sum(),
                     x['count'].sum(),
                     x['count'].sum()/x['region_area_mm2'].sum(),
                     x['density_mm2'].mean(),
                     x['density_mm2'].std(),
                     x['density_mm2'].std()/np.sqrt(len(x['density_mm2'])),
                     len(x['density_mm2'])
                    ]
                )))
            ).reset_index()
        cnts = cnts.merge(fc,on=mergeon)
        cnts['measured_frame_count'] = cnts['measured_frame_count'].astype(int)
        # get fractions also
        totals = cnts.groupby(mergeon).sum()[['cummulative_count']].\
            rename(columns={'cummulative_count':'sample_total_count'}).reset_index()
        cnts = cnts.merge(totals,on=mergeon)
        cnts['fraction'] = cnts.apply(lambda x: x['cummulative_count']/x['sample_total_count'],1)
        return cnts
    def project_counts(self,subsets=None):
        mergeon = self.cdf.project_columns+['region_label']
        cnts = self.sample_counts(subsets=subsets).groupby(mergeon+['phenotype_label']).\
            apply(lambda x: 
                pd.Series(dict(zip(
                    ['cummulative_count'],
                    [x['cummulative_count'].sum()]
                )))
            ).reset_index()
        totals = cnts.groupby(mergeon).sum()[['cummulative_count']].\
            rename(columns={'cummulative_count':'project_total_count'})
        cnts = totals.merge(cnts,on=mergeon)
        cnts['fraction'] = cnts.apply(lambda x: x['cummulative_count']/x['project_total_count'],1)
        return cnts
