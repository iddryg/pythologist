from pythologist.selection import SubsetLogic as SL
import pandas as pd
import numpy as np
import math
from pythologist.measurements import Measurement
from collections import namedtuple

PercentageLogic = namedtuple('PercentageLogic',('numerator','denominator','label'))

class Counts(Measurement):
    @staticmethod
    def _preprocess_dataframe(cdf,*args,**kwargs):
        # set our phenotype labels
        data = pd.DataFrame(cdf) # we don't need to do anything special with the dataframe for counting
        data['phenotype_label'] = data.apply(lambda x: 
                [k for k,v in x['phenotype_calls'].items() if v==1]
            ,1).apply(lambda x: np.nan if len(x)==0 else x[0])
        return data
    def frame_counts(self,subsets=None,_apply_filter=True):
        """
        Frame counts is the core of all the counting operations.  It counts on a per-frame/per-region basis.

        Args:
            subsets (list): a list of Subset Objects.  if not specified, the phenotypes are used.
            _apply_filter (bool): specify whether or not to apply the pixel and percent filter.  sample_counts uses this to defer application of the filter till the end.

        Returns:
            pandas.DataFrame: A dataframe of count data
        """
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
            seen_labels = []
            for sl in subsets:
                if sl.label in seen_labels: raise ValueError("cannot use the same label twice in the subsets list")
                seen_labels.append(sl.label)

                df = self.cdf.subset(sl)
                df = df.groupby(mergeon).count()[['cell_index']].\
                    rename(columns={'cell_index':'count'}).reset_index()
                df = self.measured_regions.merge(df,on=mergeon,how='left').fillna(0)
                df['phenotype_label'] = sl.label
                cnts.append(df)
            cnts = pd.concat(cnts)
        cnts = cnts[mergeon+['region_area_pixels','phenotype_label','count']]
        cnts['region_area_mm2'] = cnts.apply(lambda x: 
            (x['region_area_pixels']/1000000)*(self.microns_per_pixel*self.microns_per_pixel),1)
        cnts['density_mm2'] = cnts.apply(lambda x: np.nan if x['region_area_mm2'] == 0 else x['count']/x['region_area_mm2'],1)

        totals = cnts.groupby(mergeon).sum()[['count']].\
            rename(columns={'count':'frame_total_count'}).reset_index()
        cnts = cnts.merge(totals,on=mergeon)
        cnts['fraction'] = cnts.apply(lambda x: np.nan if x['frame_total_count']==0 else x['count']/x['frame_total_count'],1)
        cnts['percent'] = cnts['fraction'].multiply(100)

        # make sure regions of size zero have counts of np.nan
        if _apply_filter:
            cnts.loc[cnts['frame_total_count']<self.minimum_denominator_count,['fraction','percent']] = np.nan
            cnts.loc[cnts['region_area_pixels']<self.minimum_region_size_pixels,['density_mm2']] = np.nan
        # Deal with the percents if we are measuring them

        cnts['count'] = cnts['count'].astype(int)

        if subsets is not None:
            # if we are doing subsets we've lost any relevent reference counts in the subsetting process
            cnts['frame_total_count'] = np.nan
            cnts['fraction'] = np.nan
            cnts['percent'] = np.nan

        return cnts

    def sample_counts(self,subsets=None):
        mergeon = self.cdf.sample_columns+['region_label']
        fc = self.measured_regions[self.cdf.frame_columns+['region_label']].drop_duplicates().groupby(mergeon).\
            count()[['frame_id']].rename(columns={'frame_id':'frame_count'}).\
            reset_index()

        # Take one pass through where we apply the minimum pixel count
        cnts1 = self.frame_counts(subsets=subsets).\
            groupby(mergeon+['phenotype_label']).\
            apply(lambda x:
                pd.Series(dict(zip(
                    [
                     #'cumulative_region_area_pixels',
                     #'cumulative_region_area_mm2',
                     #'cumulative_count',
                     #'cumulative_density_mm2',
                     'mean_density_mm2',
                     'stddev_density_mm2',
                     'stderr_density_mm2',
                     'measured_frame_count'
                    ],
                    [
                     #x['region_area_pixels'].sum(),
                     #x['region_area_mm2'].sum(),
                     #x['count'].sum(),
                     #np.nan if x['region_area_mm2'].sum() == 0 else x['count'].sum()/x['region_area_mm2'].sum(),
                     x['density_mm2'].mean(),
                     x['density_mm2'].std(),
                     x['density_mm2'].std()/np.sqrt(len([y for y in x['density_mm2'] if y==y])),
                     len([y for y in x['density_mm2'] if y==y])
                    ]
                )))
            ).reset_index()
        cnts1= cnts1.merge(fc,on=mergeon)
        #cnts1['measured_frame_count'] = cnts1['measured_frame_count'].astype(int)

        # Take one pass through ignoring the minimum pixel count at the frame level and applying it to the whole sample for cumulative measures
        cnts2 = self.frame_counts(subsets=subsets,_apply_filter=False).\
            groupby(mergeon+['phenotype_label']).\
            apply(lambda x:
                pd.Series(dict(zip(
                    [
                     'cumulative_region_area_pixels',
                     'cumulative_region_area_mm2',
                     'cumulative_count',
                     'cumulative_density_mm2',
                     #'mean_density_mm2',
                     #'stddev_density_mm2',
                     #'stderr_density_mm2',
                     #'measured_frame_count'
                    ],
                    [
                     x['region_area_pixels'].sum(),
                     x['region_area_mm2'].sum(),
                     x['count'].sum(),
                     np.nan if x['region_area_mm2'].sum() == 0 else x['count'].sum()/x['region_area_mm2'].sum(),
                     #x['density_mm2'].mean(),
                     #x['density_mm2'].std(),
                     #x['density_mm2'].std()/np.sqrt(len([y for y in x['density_mm2'] if y==y])),
                     #len([y for y in x['density_mm2'] if y==y])
                    ]
                )))
            ).reset_index()
        cnts2= cnts2.merge(fc,on=mergeon)
        cnts = cnts2.merge(cnts1,on=mergeon+['phenotype_label','frame_count'])


        # get fractions also
        totals = cnts.groupby(mergeon).sum()[['cumulative_count']].\
            rename(columns={'cumulative_count':'sample_total_count'}).reset_index()
        cnts = cnts.merge(totals,on=mergeon)
        cnts['fraction'] = cnts.apply(lambda x: np.nan if x['sample_total_count']==0 else x['cumulative_count']/x['sample_total_count'],1)

        cnts['percent'] = cnts['fraction'].multiply(100)

        cnts['measured_frame_count'] = cnts['measured_frame_count'].astype(int)

        cnts.loc[cnts['cumulative_region_area_pixels']<self.minimum_region_size_pixels,['cumulative_density_mm2']] = np.nan
        cnts.loc[cnts['sample_total_count']<self.minimum_denominator_count,['percent','fraction']] = np.nan
        cnts['cumulative_count'] = cnts['cumulative_count'].astype(int)
        if subsets is not None:
            # if we are doing subsets we've lost any relevent reference counts in the subsetting process
            cnts['sample_total_count'] = np.nan
            cnts['fraction'] = np.nan
            cnts['percent'] = np.nan
        return cnts

    def project_counts(self,subsets=None):
        #raise VaueError("This function has not been tested in the current build.\n")
        #mergeon = self.cdf.project_columns+['region_label']

        pjt = self.sample_counts(subsets=subsets).groupby(['project_id',
                              'project_name',
                              'region_label',
                              'phenotype_label'])[['cumulative_count',
                                                   'cumulative_region_area_pixels',
                                                   'cumulative_region_area_mm2',
                                                  ]].sum()
        pjt['cumulative_density_mm2'] = pjt.apply(lambda x: np.nan if x['cumulative_region_area_mm2']==0 else x['cumulative_count']/x['cumulative_region_area_mm2'],1)
        pjt = pjt.reset_index()
        tot = pjt.groupby(['project_id','project_name','region_label']).sum()[['cumulative_count']].\
            rename(columns={'cumulative_count':'project_total_count'}).reset_index()
        pjt = pjt.merge(tot,on=['project_id','project_name','region_label'])
        pjt['fraction'] = pjt.apply(lambda x: x['cumulative_count']/x['project_total_count'],1)
        pjt['percent'] = pjt['fraction'].multiply(100)
        if subsets is not None:
            cnts['project_total_count'] = np.nan
            cnts['fraction'] = np.nan
            cnts['percent'] = np.nan
        return pjt


    def frame_percentages(self,percentage_logic_list):
        criteria = self.cdf.frame_columns+['region_label']
        results = []
        seen_labels = []
        for entry in percentage_logic_list:
            if entry.label in seen_labels: raise ValueError("cannot use the same label twice in the percentage logic list")
            seen_labels.append(entry.label)
            entry.numerator.label = 'numerator'
            entry.denominator.label = 'denominator'
            numerator = self.frame_counts(subsets=[entry.numerator])
            denominator = self.frame_counts(subsets=[entry.denominator])
            numerator = numerator[criteria+['count']].rename(columns={'count':'numerator'})
            denominator = denominator[criteria+['count']].rename(columns={'count':'denominator'})
            combo = numerator.merge(denominator,on=criteria, how='outer')
            combo['fraction'] = combo.\
                apply(lambda x: np.nan if x['denominator']<self.minimum_denominator_count else x['numerator']/x['denominator'],1)
            combo['percent'] = combo['fraction'].multiply(100)
            combo['phenotype_label'] = entry.label
            results.append(combo)
        return pd.concat(results)
    def sample_percentages(self,percentage_logic_list):
        mergeon = self.cdf.sample_columns+['region_label']

        fc = self.measured_regions[self.cdf.frame_columns+['region_label']].drop_duplicates().groupby(mergeon).\
            count()[['frame_id']].rename(columns={'frame_id':'frame_count'}).\
            reset_index()


        # Do this with filtering for the mean stderr versions
        fp = self.frame_percentages(percentage_logic_list)
        cnts = fp.groupby(self.cdf.sample_columns+['phenotype_label','region_label']).\
           apply(lambda x:
           pd.Series(dict({
               'cumulative_numerator':x['numerator'].sum(),
               'cumulative_denominator':x['denominator'].sum(),
               'cumulative_fraction':np.nan if x['denominator'].sum()!=x['denominator'].sum() or x['denominator'].sum()<self.minimum_denominator_count else x['numerator'].sum()/x['denominator'].sum(),
               'cumulative_percent':np.nan if x['denominator'].sum()!=x['denominator'].sum() or x['denominator'].sum()<self.minimum_denominator_count else 100*x['numerator'].sum()/x['denominator'].sum(),
               'mean_fraction':x['fraction'].mean(),
               'stdev_fraction':x['fraction'].std(),
               'stderr_fraction':np.nan if len([y for y in x['fraction'] if y==y])<=1 else x['fraction'].std()/np.sqrt(len([y for y in x['fraction'] if y==y])),
               'mean_percent':x['percent'].mean(),
               'stdev_percent':x['percent'].std(),
               'stderr_percent':np.nan if len([y for y in x['percent'] if y==y])<=1 else x['percent'].std()/np.sqrt(len([y for y in x['percent'] if y==y])),
               'measured_frame_count':len([y for y in x['fraction'] if y==y]),
           }))
           ).reset_index()
        cnts = cnts.merge(fc,on=mergeon)
        cnts['measured_frame_count'] = cnts['measured_frame_count'].astype(int)

        return cnts


