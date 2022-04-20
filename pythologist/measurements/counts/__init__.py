from pythologist.selection import SubsetLogic as SL
import pandas as pd
import numpy as np
import math
from pythologist.measurements import Measurement
from collections import namedtuple
from scipy.stats import sem

_degrees_of_freedom = 1

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
            #cnts = self.groupby(mergeon+['phenotype_label']).count()[['cell_index']].\
            #    rename(columns={'cell_index':'count'})
            cnts = self.groupby(mergeon+['phenotype_label']).\
                apply(lambda x: pd.Series(dict(zip(
                    ['count','cell_area_pixels'],
                    [len(x['cell_index']),sum(x['cell_area'])]
                )))).reset_index()
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
                #df = df.groupby(mergeon).count()[['cell_index']].\
                #    rename(columns={'cell_index':'count'}).reset_index()
                if df.shape[0] > 0:
                    df = df.groupby(mergeon).\
                    apply(lambda x: pd.Series(dict(zip(
                        ['count','cell_area_pixels'],
                        [len(x['cell_index']),sum(x['cell_area'])]
                    )))).reset_index()
                else:
                    df = self.measured_regions[mergeon]
                    df.loc[:,'count'] = 0
                    df.loc[:,'cell_area_pixels'] = 0
                df = self.measured_regions.merge(df,on=mergeon,how='left').fillna(0)
                df['phenotype_label'] = sl.label
                cnts.append(df)
            cnts = pd.concat(cnts)
        cnts = cnts[mergeon+['region_area_pixels','phenotype_label','count','cell_area_pixels']]
        cnts['region_area_mm2'] = cnts.apply(lambda x: 
            (x['region_area_pixels']/1000000)*(self.microns_per_pixel*self.microns_per_pixel),1)
        cnts['density_mm2'] = cnts.apply(lambda x: np.nan if x['region_area_mm2'] == 0 else x['count']/x['region_area_mm2'],1)

        totals = cnts.groupby(mergeon).sum()[['count']].\
            rename(columns={'count':'frame_total_count'}).reset_index()
        cnts = cnts.merge(totals,on=mergeon)
        cnts['population_percent'] = cnts.apply(lambda x: np.nan if x['frame_total_count']==0 else 100*x['count']/x['frame_total_count'],1)
        cnts['area_coverage_percent'] = cnts.apply(lambda x: np.nan if x['region_area_pixels'] <  self.minimum_region_size_pixels else 100*x['cell_area_pixels']/x['region_area_pixels'],1)

        # make sure regions of size zero have counts of np.nan
        if _apply_filter:
            cnts.loc[cnts['frame_total_count']<self.minimum_denominator_count,['fraction_total_count','population_percent']] = np.nan
            cnts.loc[cnts['region_area_pixels']<self.minimum_region_size_pixels,['density_mm2']] = np.nan
        # Deal with the percents if we are measuring them

        cnts['count'] = cnts['count'].astype(int)

        if subsets is not None:
            # if we are doing subsets we've lost any relevent reference counts in the subsetting process
            cnts['frame_total_count'] = np.nan
            cnts['population_percent'] = np.nan

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
                     'mean_density_mm2',
                     'stddev_density_mm2',
                     'stderr_density_mm2',
                     'mean_area_coverage_percent',
                     'stddev_area_coverage_percent',
                     'stderr_area_coverage_percent',
                     'measured_frame_count'
                    ],
                    [
                     x['density_mm2'].mean(),
                     np.nan if len([y for y in x['density_mm2'] if y==y]) <=1 else x['density_mm2'].std(ddof=_degrees_of_freedom,skipna=True),
                     np.nan if len([y for y in x['density_mm2'] if y==y]) <=1 else sem(x['density_mm2'],ddof=_degrees_of_freedom,nan_policy='omit'),
                     x['area_coverage_percent'].mean(),
                     np.nan if len([y for y in x['area_coverage_percent'] if y==y]) <=1 else x['area_coverage_percent'].std(ddof=_degrees_of_freedom,skipna=True),
                     np.nan if len([y for y in x['area_coverage_percent'] if y==y]) <=1 else sem(x['area_coverage_percent'],ddof=_degrees_of_freedom,nan_policy='omit'),
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
                     'cumulative_cell_area_pixels',
                     'cumulative_area_coverage_percent'
                    ],
                    [
                     x['region_area_pixels'].sum(),
                     x['region_area_mm2'].sum(),
                     x['count'].sum(),
                     np.nan if x['region_area_pixels'].sum() < self.minimum_region_size_pixels else x['count'].sum()/x['region_area_mm2'].sum(),
                     x['cell_area_pixels'].sum(),
                     np.nan if x['region_area_pixels'].sum() < self.minimum_region_size_pixels else 100*x['cell_area_pixels'].sum()/x['region_area_pixels'].sum()
                    ]
                )))
            ).reset_index()
        cnts2= cnts2.merge(fc,on=mergeon)
        cnts = cnts2.merge(cnts1,on=mergeon+['phenotype_label','frame_count'])


        # get fractions also
        totals = cnts.groupby(mergeon).sum()[['cumulative_count']].\
            rename(columns={'cumulative_count':'sample_total_count'}).reset_index()
        cnts = cnts.merge(totals,on=mergeon)
        cnts['population_percent'] = cnts.apply(lambda x: np.nan if x['sample_total_count']==0 else 100*x['cumulative_count']/x['sample_total_count'],1)

        cnts['measured_frame_count'] = cnts['measured_frame_count'].astype(int)

        cnts.loc[cnts['cumulative_region_area_pixels']<self.minimum_region_size_pixels,['cumulative_density_mm2']] = np.nan
        cnts.loc[cnts['sample_total_count']<self.minimum_denominator_count,['population_percent']] = np.nan
        cnts['cumulative_count'] = cnts['cumulative_count'].astype(int)
        if subsets is not None:
            # if we are doing subsets we've lost any relevent reference counts in the subsetting process
            cnts['sample_total_count'] = np.nan
            cnts['population_percent'] = np.nan

        cnts['cumulative_region_area_pixels'] = cnts['cumulative_region_area_pixels'].astype(int)
        cnts['cumulative_cell_area_pixels'] = cnts['cumulative_cell_area_pixels'].astype(int)
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
        pjt['population_percent'] = pjt.apply(lambda x: np.nan if x['project_total_count']==0 else 100*x['cumulative_count']/x['project_total_count'],1)
        if subsets is not None:
            cnts['project_total_count'] = np.nan
            cnts['population_percent'] = np.nan
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
            combo['percent'] = combo.\
                apply(lambda x: np.nan if x['denominator']<self.minimum_denominator_count else 100*x['numerator']/x['denominator'],1)
            combo['phenotype_label'] = entry.label
            results.append(combo)
        df = pd.concat(results)
        df['qualified_percent'] = df['denominator'].apply(lambda x: x>=self.minimum_denominator_count)
        return df
    def sample_percentages(self,percentage_logic_list):
        #mergeon = self.cdf.sample_columns+['region_label']

        #fc = self.measured_regions[self.cdf.frame_columns+['region_label']].drop_duplicates().groupby(mergeon).\
        #    count()[['frame_id']].rename(columns={'frame_id':'frame_count'}).\
        #    reset_index()

        # Get observed regions
        #fo = self.measured_regions[self.cdf.sample_columns+['region_label']].drop_duplicates()
        # Get sample count
        fc = self.measured_regions[self.cdf.frame_columns+['region_label']].drop_duplicates().groupby(self.cdf.sample_columns+['region_label']).\
            count()[['frame_id']].rename(columns={'frame_id':'measured_frame_count'}).\
            reset_index()
        #fc = fo.merge(fc,on=self.cdf.sample_columns,how='left').fillna(0)
        fpheno = pd.DataFrame({'phenotype_label':[x.label for x in percentage_logic_list]})
        fpheno['_key'] = 1
        fc['_key'] = 1
        fc = fc.merge(fpheno,on=['_key']).drop(columns=['_key'])

        #sp = self.sample_percentages(percentage_logic_list)
        # Get measured sample counts
        #msc = sp[['project_id','project_name','sample_id','sample_name','region_label','cumulative_denominator','phenotype_label']].\
        #    drop_duplicates()
        #msc = msc.loc[msc['cumulative_denominator']>=self.minimum_denominator_count].drop_duplicates().\
        #    groupby(['project_id','project_name','region_label','phenotype_label']).count()[['sample_id']].\
        #    reset_index().\
        #    rename(columns={'sample_id':'measured_sample_count'})
        #sc = sc.merge(msc,on=['project_id','project_name','region_label','phenotype_label'],how='left').fillna(0)
        #sc['measured_sample_count'] = sc['measured_sample_count'].astype(int)

        # Do this with filtering for the mean stderr versions
        fp = self.frame_percentages(percentage_logic_list)
        mfc = fp[self.cdf.frame_columns+['region_label','denominator','phenotype_label']].\
            drop_duplicates()
        mfc = mfc.loc[mfc['denominator']>=self.minimum_denominator_count].drop_duplicates().\
            groupby(self.cdf.sample_columns+['region_label','phenotype_label']).count()[['frame_id']].\
            reset_index().\
            rename(columns={'frame_id':'qualified_frame_count'})
        #print(fc.columns)
        #print(mfc.columns)
        fc = fc.merge(mfc,on=self.cdf.sample_columns+['region_label','phenotype_label'],how='left').fillna(0)


        cnts = fp.groupby(self.cdf.sample_columns+['phenotype_label','region_label']).\
           apply(lambda x:
           pd.Series(dict({
               'cumulative_numerator':x['numerator'].sum(),
               'cumulative_denominator':x['denominator'].sum(),
               'cumulative_percent':np.nan if x['denominator'].sum()!=x['denominator'].sum() or x['denominator'].sum()<self.minimum_denominator_count else 100*x['numerator'].sum()/x['denominator'].sum(),
               'mean_percent':x['percent'].mean(),
               'stddev_percent':np.nan if len([y for y in x['percent'] if y==y]) <= 1 else x['percent'].std(ddof=_degrees_of_freedom,skipna=True),
               'stderr_percent':np.nan if len([y for y in x['percent'] if y==y]) <= 1 else sem(x['percent'],ddof=_degrees_of_freedom,nan_policy='omit'),
               #'measured_frame_count':len([y for y in x['percent'] if y==y]),
           }))
           ).reset_index()
        cnts = cnts.merge(fc,on=self.cdf.sample_columns+['region_label','phenotype_label'])
        cnts['qualified_frame_count'] = cnts['qualified_frame_count'].astype(int)
        cnts['cumulative_numerator'] = cnts['cumulative_numerator'].astype(int)
        cnts['cumulative_denominator'] = cnts['cumulative_denominator'].astype(int)
        cnts['qualified_cumulative_percent'] = cnts['cumulative_denominator'].apply(lambda x: x>=self.minimum_denominator_count)
        #stc = fp.groupby(self.cdf.sample_columns+['region_label']).sum()[['denominator']]
        #cnts['sample_total_count'] = cnts['sample_total_count'].astype(int)

        # add frame_counts
        _framecounts=self.measured_regions[['sample_name','frame_id']].\
            drop_duplicates().groupby(['sample_name']).count()[['frame_id']].\
            rename(columns={'frame_id':'frame_count'}).reset_index()
        cnts = cnts.merge(_framecounts,on=['sample_name'])

        return cnts
    def project_percentages(self,percentage_logic_list):
        #mergeon = self.cdf.project_columns+['phenotype_label','region_label']

        # Get observed regions
        #so = self.measured_regions[self.cdf.project_columns+['region_label']].drop_duplicates()
        # Get sample count

        sc = self.measured_regions[self.cdf.sample_columns+['region_label']].drop_duplicates().groupby(self.cdf.project_columns+['region_label']).\
            count()[['sample_id']].rename(columns={'sample_id':'measured_sample_count'}).\
            reset_index()
        #sc = so.merge(sc,on=self.cdf.project_columns,how='left').fillna(0)
        spheno = pd.DataFrame({'phenotype_label':[x.label for x in percentage_logic_list]})
        spheno['_key'] = 1
        sc['_key'] = 1
        sc = sc.merge(spheno,on=['_key']).drop(columns=['_key'])

        sp = self.sample_percentages(percentage_logic_list)
        # Get measured sample counts
        msc = sp[['project_id','project_name','sample_id','sample_name','region_label','cumulative_denominator','phenotype_label']].\
            drop_duplicates()
        msc = msc.loc[msc['cumulative_denominator']>=self.minimum_denominator_count].drop_duplicates().\
            groupby(['project_id','project_name','region_label','phenotype_label']).count()[['sample_id']].\
            reset_index().\
            rename(columns={'sample_id':'qualified_sample_count'})
        sc = sc.merge(msc,on=['project_id','project_name','region_label','phenotype_label'],how='left').fillna(0)
        sc['qualified_sample_count'] = sc['qualified_sample_count'].astype(int)

        pp = sp.groupby(self.cdf.project_columns+['phenotype_label','region_label']).sum()\
            [['cumulative_numerator','cumulative_denominator']].reset_index()
        pp['cumulative_percent'] = pp.apply(lambda x: np.nan if x['cumulative_denominator']<self.minimum_denominator_count else 100*x['cumulative_numerator']/x['cumulative_denominator'],1)
        pp = pp.merge(sc,on=self.cdf.project_columns+['region_label','phenotype_label'])
        pp['cumulative_numerator'] = pp['cumulative_numerator'].astype(int)
        pp['cumulative_denominator'] = pp['cumulative_denominator'].astype(int)
        pp['qualified_cumulative_percent'] = pp['cumulative_denominator'].apply(lambda x: x>=self.minimum_denominator_count)

        return pp

