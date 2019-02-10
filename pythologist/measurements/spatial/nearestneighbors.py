import pandas as pd
import sys
from pythologist.measurements import Measurement
import numpy as np
from scipy.spatial.distance import cdist
class NearestNeighbors(Measurement):
    @staticmethod
    def _preprocess_dataframe(cdf,*args,**kwargs):
        #mergeon = ['project_id','sample_id','frame_id','cell_index']
        mr = cdf.get_measured_regions().drop(columns='region_area_pixels')
        cdf['phenotype_label'] = cdf.apply(lambda x: 
                [k for k,v in x['phenotype_calls'].items() if v==1]
            ,1).apply(lambda x: np.nan if len(x)==0 else x[0])
        subsets = []
        for i,r in mr.iterrows():
            if 'verbose' in kwargs and kwargs['verbose'] is True: 
                sys.stderr.write(str(i+1)+'/'+str(mr.shape[0])+"\n")
            pts = cdf[(cdf['frame_id']==r['frame_id'])&\
                (cdf['frame_name']==r['frame_name'])&\
                (cdf['sample_id']==r['sample_id'])&\
                (cdf['sample_name']==r['sample_name'])&\
                (cdf['project_id']==r['project_id'])&\
                (cdf['project_name']==r['project_name'])&\
                (cdf['region_label']==r['region_label'])
            ].copy()
            pts = pts.dropna(subset=['phenotype_label']).\
                 drop(columns=['regions','scored_calls','phenotype_calls',
                       'channel_values','cell_area','edge_length','neighbors'])
            pts['coord'] = pts.apply(lambda x: (x['x'],x['y']),1)
            pts = pts.reset_index(drop=True)
            phenos = pts['phenotype_label'].unique()
            distance = pd.DataFrame(cdist(list(pts['coord']),list(pts['coord'])))
            for i in range(0,distance.shape[0]): distance[i][i] = np.nan
            for pheno1 in phenos:
                pp1 = pts[pts['phenotype_label']==pheno1]
                for pheno2 in phenos:
                    #if pheno1 == pheno2: continue
                    pp2 = pts[pts['phenotype_label']==pheno2]            
                    subset = pd.DataFrame(distance.loc[pp1.index,pp2.index].apply(lambda x: x.min(),1)).\
                        merge(pts,left_index=True,right_index=True).rename(columns={0:'distance'}).\
                        drop(columns='coord')
                    subset['neighbor_phenotype_label'] = pheno2
                    subsets.append(subset)
        subsets = pd.concat(subsets)
        return subsets
    def _distance(self,mergeon,minimum_edges):
        mr = self.measured_regions[mergeon].drop_duplicates().copy()
        mr['_key'] = 1
        mp = pd.DataFrame({'phenotype_label':self.measured_phenotypes})
        mp['_key'] = 1
        mn = pd.DataFrame({'neighbor_phenotype_label':self.measured_phenotypes})
        mn['_key'] = 1
        data = mr.merge(mp,on='_key').merge(mn,on='_key').drop(columns='_key')
        fdata = self.groupby(mergeon+['phenotype_label','neighbor_phenotype_label']).\
            apply(lambda x: 
                pd.Series(dict(zip(
                    ['edge_count',
                     'mean_distance_pixels',
                     'mean_distance_um',
                     'stddev_distance_pixels',
                     'stddev_distance_um',
                     'stderr_distance_pixels',
                     'stderr_distance_um'
                    ],
                    [
                      len(x['distance']),
                      x['distance'].mean(),
                      x['distance'].mean()*self.microns_per_pixel,
                      x['distance'].std(),
                      x['distance'].std()*self.microns_per_pixel,
                      x['distance'].std()/np.sqrt(len(x['distance'])),
                      x['distance'].std()*self.microns_per_pixel/np.sqrt(len(x['distance']))
                    ]
           )))
        ).reset_index()
        fdata.loc[fdata['edge_count']<minimum_edges,'mean_distance_pixels'] = np.nan
        fdata.loc[fdata['edge_count']<minimum_edges,'mean_distance_um'] = np.nan
        fdata.loc[fdata['edge_count']<minimum_edges,'stddev_distance_pixels'] = np.nan
        fdata.loc[fdata['edge_count']<minimum_edges,'stddev_distance_um'] = np.nan
        fdata.loc[fdata['edge_count']<minimum_edges,'stderr_distance_pixels'] = np.nan
        fdata.loc[fdata['edge_count']<minimum_edges,'stderr_distance_um'] = np.nan
        data = data.merge(fdata,on=list(data.columns),how='left')
        data['minimum_edges'] = minimum_edges
        return data
    def frame_distance(self,minimum_edges=20):
        mergeon=['project_id','project_name','sample_id','sample_name','frame_id','frame_name','region_label']
        return self._distance(mergeon,minimum_edges)
    def _cummulative_sample_distance(self,minimum_edges=20):
        mergeon=['project_id','project_name','sample_id','sample_name','region_label']
        data = self._distance(mergeon,minimum_edges).\
            rename(columns={'edge_count':'cummulative_edge_count',
                            'mean_distance_pixels':'mean_cummulative_distance_pixels',
                            'mean_distance_um':'mean_cummulative_distance_um',
                            'stddev_distance_pixels':'stddev_cummulative_distance_pixels',
                            'stddev_distance_um':'stddev_cummulative_distance_um',
                            'stddev_distance_pixels':'stddev_cummulative_distance_pixels',
                            'stderr_distance_um':'stddev_cummulative_distance_um',
                           })
        return data
    def _mean_sample_distance(self,minimum_edges=20):
        mergeon=['project_id','project_name','sample_id','sample_name','region_label']
        mr = self.measured_regions[mergeon+['frame_id','frame_name']].drop_duplicates().copy()
        mr = mr.groupby(mergeon).count()[['frame_id']].rename(columns={'frame_id':'frame_count'}).\
            reset_index()
        mr['_key'] = 1
        mp = pd.DataFrame({'phenotype_label':self.measured_phenotypes})
        mp['_key'] = 1
        mn = pd.DataFrame({'neighbor_phenotype_label':self.measured_phenotypes})
        mn['_key'] = 1
        blank = mr.merge(mp,on='_key').merge(mn,on='_key').drop(columns='_key')

        data = self.frame_distance(minimum_edges).dropna()
        data = data.groupby(mergeon+['phenotype_label','neighbor_phenotype_label']).\
            apply(lambda x:
                pd.Series(dict(zip(
                    ['mean_mean_distance_pixels',
                     'mean_mean_distance_um',
                     'stddev_mean_distance_pixels',
                     'stddev_mean_distance_um',
                     'stderr_mean_distance_pixels',
                     'stderr_mean_distance_um',
                     'measured_frame_count'
                    ],
                    [
                      x['mean_distance_pixels'].mean(),
                      x['mean_distance_um'].mean(),
                      x['mean_distance_pixels'].std(),
                      x['mean_distance_um'].std(),
                      x['mean_distance_pixels'].std()/np.sqrt(len(x['mean_distance_pixels'])),
                      x['mean_distance_um'].std()/np.sqrt(len(x['mean_distance_pixels'])),
                      len(x['mean_distance_pixels'])
                    ]
                )))
            ).reset_index()
        data = blank.merge(data,on=mergeon+['phenotype_label','neighbor_phenotype_label'],how='left')
        return data
    def sample_distance(self,minimum_edges=20):
        mergeon=['project_id','project_name','sample_id','sample_name','region_label']
        v1 = self._cummulative_sample_distance(minimum_edges)
        v2 = self._mean_sample_distance(minimum_edges)
        data = v1.merge(v2,on=mergeon+['phenotype_label','neighbor_phenotype_label'])
        data.loc[data['measured_frame_count'].isna(),'measured_frame_count'] = 0
        return data

    def frame_proximity(self,threshold_um,phenotype):
        threshold  = threshold_um/self.microns_per_pixel
        mergeon = ['project_id','project_name','sample_id','sample_name',
               'frame_id','frame_name','region_label'
              ]
        df = self.loc[(self['neighbor_phenotype_label']==phenotype)
                 ].copy()
        df.loc[df['distance']>=threshold,'location'] = 'far'
        df.loc[df['distance']<threshold,'location'] = 'near'
        df = df.groupby(mergeon+['phenotype_label','neighbor_phenotype_label','location']).count()[['cell_index']].\
            rename(columns={'cell_index':'count'}).reset_index()[mergeon+['phenotype_label','location','count']]
        mr = self.measured_regions[mergeon].copy()
        mr['_key'] = 1
        mp = pd.DataFrame({'phenotype_label':self.measured_phenotypes})
        mp['_key'] = 1
        total = df.groupby(mergeon+['location']).sum()[['count']].rename(columns={'count':'total'}).reset_index()
        blank = mr.merge(mp,on='_key').merge(total,on=mergeon).drop(columns='_key')
        df = blank.merge(df,on=mergeon+['location','phenotype_label'],how='left')
        df.loc[(~df['total'].isna())&(df['count'].isna()),'count'] =0
        df['fraction'] = df.apply(lambda x: x['count']/x['total'],1)
        df = df.sort_values(mergeon+['location','phenotype_label'])
        return df
    def sample_proximity(self,threshold_um,phenotype):
        mergeon = ['project_id','project_name','sample_id','sample_name','region_label']
        fp = self.frame_proximity(threshold_um,phenotype)
        cnt = fp.groupby(mergeon+['phenotype_label','location']).sum()[['count']].reset_index()
        total = cnt.groupby(mergeon+['location']).sum()[['count']].rename(columns={'count':'total'}).\
             reset_index()
        cnt = cnt.merge(total,on=mergeon+['location']).sort_values(mergeon+['location','phenotype_label'])
        cnt['fraction'] = cnt.apply(lambda x: x['count']/x['total'],1)
        return cnt
