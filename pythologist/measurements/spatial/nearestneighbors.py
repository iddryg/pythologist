import pandas as pd
import sys
from pythologist.measurements import Measurement
import numpy as np
from scipy.spatial.distance import cdist
class NearestNeighbors(Measurement):
    @staticmethod
    def _preprocess_dataframe(cdf,*args,**kwargs):
        def _mindist_nodiag(pts1,pts2):
            mat = cdist(list(pts1),list(pts2))
            if len(pts1)==len(pts2) and set(pts1.index) == set(pts2.index): 
                np.fill_diagonal(mat,np.nan)
            matmin = np.nanargmin(mat,axis=1)
            data = [(pts1.index[i],pts1.iloc[i],pts2.index[y],pts2.iloc[y],mat[i,y]) for i,y in enumerate(matmin)]
            data = pd.DataFrame(data,columns=['cell_index','cell_coord','neighbor_cell_index','neighbor_cell_coord','minimum_distance_pixels'])
            return data
        def _combine_dfs(minima,index,index_names):
            n1 = minima
            n1['_key'] = 1
            n2 = pd.DataFrame(index,index=index_names).T
            n2['_key'] = 1
            return n2.merge(n1,on='_key').drop(columns='_key')
        cdf = cdf.copy()
        if 'verbose'  in kwargs and kwargs['verbose']: sys.stderr.write("read phenotype label\n")
        mr = cdf.get_measured_regions().drop(columns='region_area_pixels')
        cdf['phenotype_label'] = cdf.apply(lambda x: 
                [k for k,v in x['phenotype_calls'].items() if v==1]
            ,1).apply(lambda x: np.nan if len(x)==0 else x[0])
        phenotypes = cdf['phenotype_label'].unique()
        if 'verbose'  in kwargs and kwargs['verbose']: sys.stderr.write("get all coordinates\n")
        cdf['coord'] = cdf.apply(lambda x: (x['x'],x['y']),1)
        if 'verbose'  in kwargs and kwargs['verbose']: sys.stderr.write("get all coord pairs\n")
        cdf = cdf.groupby(list(mr.columns)+['phenotype_label']).apply(lambda x: 
            pd.Series(dict(zip(
                ['cell_index','coordinates'],
                [list(x['cell_index']),list(x['coord'])]            
            )))
        ).reset_index()
        if 'verbose'  in kwargs and kwargs['verbose']: sys.stderr.write("set up comparisons points\n")
        cdf = cdf.merge(cdf.rename(columns={'cell_index':'neighbor_cell_index',
                                            'coordinates':'neighbor_coordinates',
                                            'phenotype_label':'neighbor_phenotype_label'}),
                       on = list(mr.columns))
        if 'verbose'  in kwargs and kwargs['verbose']: sys.stderr.write("get minima\n")
        cdf = cdf.set_index(list(mr.columns)+['phenotype_label','neighbor_phenotype_label']).\
            apply(lambda x: 
                    _mindist_nodiag(pd.Series(x['coordinates'],index=x['cell_index']),
                                    pd.Series(x['neighbor_coordinates'],index=x['neighbor_cell_index']))
            ,1)
        inames = cdf.index.names
        cdf  = cdf.reset_index().rename(columns={0:'cdist'}).set_index(inames)
        if 'verbose'  in kwargs and kwargs['verbose']: sys.stderr.write("combine data\n")
        cdf = cdf.apply(lambda x: _combine_dfs(x['cdist'],x.name,cdf.index.names),1)
        return pd.concat(cdf.tolist())    
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
    def project_proximity(self,threshold_um,phenotype):
        mergeon = ['project_id','project_name','region_label']
        fp = self.sample_proximity(threshold_um,phenotype)
        cnt = fp.groupby(mergeon+['phenotype_label','location']).sum()[['count']].reset_index()
        total = cnt.groupby(mergeon+['location']).sum()[['count']].rename(columns={'count':'total'}).\
             reset_index()
        cnt = cnt.merge(total,on=mergeon+['location']).sort_values(mergeon+['location','phenotype_label'])
        cnt['fraction'] = cnt.apply(lambda x: x['count']/x['total'],1)
        return cnt
