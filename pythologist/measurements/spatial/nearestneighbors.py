import pandas as pd
import sys
from pythologist.measurements import Measurement
import numpy as np
from sklearn.neighbors import KDTree

def _clean_neighbors(left,right,k_neighbors):
    def _get_coords(cell_index,x,y):
        return pd.Series(dict(zip(
            ['cell_index','neighbor_cell_coord'],
            [cell_index,(x,y)]
        )))
    rcoords = right.apply(lambda x: 
         _get_coords(x['cell_index'],x['x'],x['y'])
    ,1).reset_index(drop=True)
    lcoords = left.apply(lambda x: 
        _get_coords(x['cell_index'],x['x'],x['y'])
    ,1).reset_index(drop=True)
    kdt = KDTree(rcoords['neighbor_cell_coord'].tolist(), leaf_size=40, metric='minkowski')
    dists, idxs = kdt.query(lcoords['neighbor_cell_coord'].tolist(),min(right.shape[0],k_neighbors+1))
    dists = pd.DataFrame(dists,index = lcoords['cell_index']).stack().reset_index().\
        rename(columns = {'level_1':'_neighbor_rank',0:'neighbor_distance_px'})
    idxs = pd.DataFrame(idxs,index = lcoords['cell_index']).stack().reset_index().\
        rename(columns = {'level_1':'_neighbor_rank',0:'neighbor_dbid'})
    dists = dists.merge(idxs,on=['_neighbor_rank','cell_index']).\
        merge(rcoords.rename(columns={'cell_index':'neighbor_cell_index'}),left_on='neighbor_dbid',right_index=True)
    dists = dists.loc[dists['cell_index']!=dists['neighbor_cell_index'],:].\
        sort_values(['cell_index','_neighbor_rank']).\
        reset_index(drop=True).drop(columns=['neighbor_dbid'])
    if dists.shape[0] == 0: return None
    _rank_code = dists.groupby('cell_index').\
        apply(lambda x:
          pd.Series(dict(zip(
              range(0,len(x['_neighbor_rank'])),
              x['_neighbor_rank']
          )))
         ).stack().reset_index().\
        rename(columns={'level_1':'neighbor_rank',0:'_neighbor_rank'})
    dists = dists.merge(_rank_code,on=['cell_index','_neighbor_rank']).drop(columns=['_neighbor_rank'])
    dists = dists.loc[dists['neighbor_rank']<k_neighbors,:] # make sure we are limited to our number
    return dists

class NearestNeighbors(Measurement):
    @staticmethod
    def _preprocess_dataframe(cdf,*args,**kwargs):
        if 'per_phenotype_neighbors' not in kwargs: raise ValueError('per_phenotype_neighbors must be defined')
        k_neighbors = kwargs['per_phenotype_neighbors']
        nn = []
        for rdf in cdf.frame_region_generator():
            if kwargs['verbose'] and rdf.shape[0]>0:
                row = rdf.iloc[0]
                sys.stderr.write("Extracting NN from "+str((row['project_id'],
                                                                    row['project_name'],
                                                                    row['sample_id'],
                                                                    row['sample_name'],
                                                                    row['frame_id'],
                                                                    row['frame_name'],
                                                                    row['region_label']
                            ))+"\n")
            for phenotype_label1 in rdf['phenotype_label'].unique():
                for phenotype_label2 in rdf['phenotype_label'].unique():
                    left = rdf.loc[rdf['phenotype_label']==phenotype_label1,:]
                    right= rdf.loc[rdf['phenotype_label']==phenotype_label2,:]
                    if left.shape[0]==0 or right.shape[0]==0: continue
                    dists = _clean_neighbors(left,right,k_neighbors)
                    if dists is None: continue
                    _df = pd.DataFrame(left[['project_id','project_name','sample_name','sample_id','frame_name','frame_id','region_label','phenotype_label','cell_index']])
                    _df['neighbor_phenotype_label'] = phenotype_label2
                    _df = _df.merge(dists,on='cell_index')
                    nn.append(_df)
        if kwargs['verbose']: sys.stderr.write("concatonating nn blocks\n")
        nn = pd.concat(nn).reset_index(drop=True)
        # add on the total rank
        def _add_index(x):
            df = pd.DataFrame({
              'overall_rank':range(0,len(x['neighbor_distance_px'])),
              'neighbor_distance_px':x['neighbor_distance_px'],
              'neighbor_cell_index':x['neighbor_cell_index']
            })
            df['project_id'] = x.name[0]
            df['sample_id'] = x.name[1]
            df['frame_id'] = x.name[2]
            df['region_label'] = x.name[3]
            df['cell_index'] = x.name[4]
            return df
        _rnks = nn.sort_values(['project_id','sample_id','frame_id','region_label','cell_index','neighbor_distance_px']).\
            reset_index(drop=True).\
            groupby(['project_id','sample_id','frame_id','region_label','cell_index']).\
            apply(lambda x: 
                _add_index(x)
                ).drop(columns='neighbor_distance_px')
        nn = nn.merge(_rnks,on=['project_id','sample_id','frame_id','region_label','cell_index','neighbor_cell_index'])
        nn['per_phenotype_neighbors'] = k_neighbors
        return nn

    def _distance(self,mergeon,minimum_edges):
        mr = self.measured_regions[mergeon].drop_duplicates().copy()
        mr['_key'] = 1
        mp = pd.DataFrame({'phenotype_label':self.measured_phenotypes})
        mp['_key'] = 1
        mn = pd.DataFrame({'neighbor_phenotype_label':self.measured_phenotypes})
        mn['_key'] = 1
        data = mr.merge(mp,on='_key').merge(mn,on='_key').drop(columns='_key')
        fdata = self.loc[self['neighbor_rank']==0].groupby(mergeon+['phenotype_label','neighbor_phenotype_label']).\
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
                      len(x['neighbor_distance_px']),
                      x['neighbor_distance_px'].mean(),
                      x['neighbor_distance_px'].mean()*self.microns_per_pixel,
                      x['neighbor_distance_px'].std(),
                      x['neighbor_distance_px'].std()*self.microns_per_pixel,
                      x['neighbor_distance_px'].std()/np.sqrt(len(x['neighbor_distance_px'])),
                      x['neighbor_distance_px'].std()*self.microns_per_pixel/np.sqrt(len(x['neighbor_distance_px']))
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
        mr = self.loc[self['neighbor_rank']==0].measured_regions[mergeon+['frame_id','frame_name']].drop_duplicates().copy()
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
        mergeon = self.cdf.frame_columns+['region_label']
        df = self.loc[(self['neighbor_phenotype_label']==phenotype)
                 ].copy()
        df.loc[df['neighbor_distance_px']>=threshold,'location'] = 'far'
        df.loc[df['neighbor_distance_px']<threshold,'location'] = 'near'
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
        mergeon = self.cdf.sample_columns+['region_label']
        fp = self.frame_proximity(threshold_um,phenotype)
        cnt = fp.groupby(mergeon+['phenotype_label','location']).sum()[['count']].reset_index()
        total = cnt.groupby(mergeon+['location']).sum()[['count']].rename(columns={'count':'total'}).\
             reset_index()
        cnt = cnt.merge(total,on=mergeon+['location']).sort_values(mergeon+['location','phenotype_label'])
        cnt['fraction'] = cnt.apply(lambda x: x['count']/x['total'],1)
        return cnt
    def project_proximity(self,threshold_um,phenotype):
        mergeon = self.cdf.project_columns+['region_label']
        fp = self.sample_proximity(threshold_um,phenotype)
        cnt = fp.groupby(mergeon+['phenotype_label','location']).sum()[['count']].reset_index()
        total = cnt.groupby(mergeon+['location']).sum()[['count']].rename(columns={'count':'total'}).\
             reset_index()
        cnt = cnt.merge(total,on=mergeon+['location']).sort_values(mergeon+['location','phenotype_label'])
        cnt['fraction'] = cnt.apply(lambda x: x['count']/x['total'],1)
        return cnt
    def threshold(self,phenotype,proximal_label,k_neighbors=1,distance_um=None,distance_pixels=None):
        if k_neighbors > self.iloc[0]['per_phenotype_neighbors']:
            raise ValueError("must select a k_neighbors smaller or equal to the min_neighbors used to generate the NearestNeighbors object")
        if phenotype not in self.cdf.phenotypes: raise ValueError("Can only threshold on one of the pre-established phenotypes (before calling nearestneighbors")
        def _add_score(d,value,label):
            d[label] = 0 if value!=value else int(value)
            return d
        if distance_um is not None and distance_pixels is None:
            distance_pixels = distance_um/self.microns_per_pixel

        nn1 = self.loc[(self['neighbor_phenotype_label']==phenotype)&\
               (self['neighbor_rank']==k_neighbors-1)
              ].copy()
        nn1['_threshold'] = np.nan
        nn1.loc[(nn1['neighbor_distance_px']<distance_pixels),'_threshold'] = 1
        if nn1.shape

        output = self.cdf.copy()
        mergeon = output.frame_columns+['region_label','cell_index']
        cdf = output.merge(nn1[mergeon+['_threshold']],on=mergeon,how='left')
        cdf['scored_calls'] = cdf.apply(lambda x:
            _add_score(x['scored_calls'],x['_threshold'],proximal_label)
        ,1)
        cdf.microns_per_pixel = self.microns_per_pixel
        return cdf.drop(columns='_threshold')
    def bin_fractions_from_neighbor(self,neighbor_phenotype,numerator_phenotypes,denominator_phenotypes,
                                         bin_size_microns=20,
                                         minimum_total_count=0,
                                         group_strategy=['project_name','sample_name']):
        # set our bin size in microns
        mynn = self.loc[self['neighbor_phenotype_label']==neighbor_phenotype].copy()
        mynn['neighbor_distance_um'] = mynn['neighbor_distance_px'].apply(lambda x: x*self.cdf.microns_per_pixel)
        rngs = np.arange(0,mynn['neighbor_distance_um'].max(),bin_size_microns)
        mynn['bins'] = pd.cut(mynn['neighbor_distance_um'],bins=rngs)
        numerator = mynn.loc[mynn['phenotype_label'].isin(numerator_phenotypes)]
        denominator = mynn.loc[mynn['phenotype_label'].isin(denominator_phenotypes)]

        numerator = numerator.groupby(group_strategy+['bins']).count()[['cell_index']].rename(columns={'cell_index':'cell_count'}).reset_index()
        numerator['group'] = 'numerator'
        denominator = denominator.groupby(group_strategy+['bins']).count()[['cell_index']].rename(columns={'cell_index':'cell_count'}).reset_index()
        denominator['group'] = 'total'
        sub = pd.concat([numerator,denominator])
        sub = sub.set_index(group_strategy+['bins']).pivot(columns='group')
        sub.columns = sub.columns.droplevel(0)
        sub = sub.reset_index()
        sub['fraction'] = sub['numerator'].divide(sub['total'])
        sub.loc[sub['numerator'].isna(),'numerator']=0
        sub.loc[sub['total'].isna(),'total']=0
        sub['right']=[int(x.right) for x in sub['bins'].tolist()]
        sub.loc[sub['total']<minimum_total_count,'fraction']=np.nan
        return sub
    def cell_proximity_cdfs(self,include_self=True,k_neighbors=50,min_neighbors=40,max_distance_px=None,max_distance_um=None):
        """
        Use the neighbors that were calculated to generate mini cell data frames based on the cells near by each cell

        Args:
            include_self (bool): Include the refernece cell in the neighborhood. default True
            k_neighbors (int): The maximum number of nearest negibhors to use.  Must be less than or equal to max_neighbors. default: 50
            max_distance_px (int): The maximum distance to constrain k_neighbors to.  default None
            max_distance_um (float): The maximum distance to constrain k_neighbors to overrides pixel setting if both are set. default None
            min_neighbors (int): Do not return cells if they do not have a local neighborhood of sufficient size

        Returns:
            Series, CellDataFrame: Returns a series from the CellDataFrame which is the cell this region is referenced from, and the CellDataFrame.
        """
        if k_neighbors > self.iloc[0]['per_phenotype_neighbors']: raise ValueError("k_neighbors must be less or equal to per_phenotype_neighbors defined earlier")
        if max_distance_um is not None:
            max_distance_px = max_distance_um/self.cdf.microns_per_pixel
        mergeon = ['project_id','project_name','sample_id','sample_name','frame_id','frame_name','region_label']
        for block in self.cdf.frame_region_generator():
            block = block.prune_neighbors()
            block_idx = block.set_index('cell_index')
            df = block.merge(self,left_on=mergeon+['cell_index'],right_on=mergeon+['neighbor_cell_index'])
            df = df.loc[df['overall_rank']<k_neighbors,:]
            if max_distance_px is not None:
                df = df.loc[df['neighbor_distance_px']<=max_distance_px,:]
            df = df.drop(columns=['neighbor_cell_index','phenotype_label_y','per_phenotype_neighbors','neighbor_cell_coord','neighbor_phenotype_label']).\
                rename(columns={'cell_index_x':'cell_index','cell_index_y':'cell_group','phenotype_label_x':'phenotype_label'})
            if include_self: 
                # We need to add on the cell itself to the cell group
                df2 = block.copy()
                df2['cell_group'] = df2['cell_index']
                df2['neighbor_distance_px'] = 0
                df2['neighbor_rank'] = -1
                df2['overall_rank'] = -1
                df = pd.concat([df,df2.loc[:,df.columns]])
            df = df.sort_values(['cell_group','cell_index']).reset_index(drop=True).set_index('cell_group')
            df.microns_per_pixel = self.cdf.microns_per_pixel
            for cell_group in df.index.unique():
                reference_cell = block_idx.loc[cell_group]
                output_cdf = df.loc[cell_group].reset_index(drop=True)
                if output_cdf.shape[0] < min_neighbors: continue
                yield reference_cell, output_cdf
    def explode_to_proximity_region_cdf(self,k_neighbors=50,max_distance_um=100,min_neighbors=40,verbose=False):
        massive = []
        for i, (ref, mdf) in enumerate(self.cell_proximity_cdfs(k_neighbors=k_neighbors,
                                                              max_distance_um=max_distance_um,
                                                              min_neighbors=min_neighbors)):
            if verbose and i%100==0: sys.stderr.write("reading block "+str(i)+"\r")
            mdf['channel_values'] = mdf['channel_values'].apply(lambda x: dict())
            mdf = mdf.rename_region(mdf.regions,str(ref['frame_id'])+'|'+str(ref['region_label'])+'|'+str(ref.name))
            massive.append(mdf)
        if verbose: sys.stderr.write("\nconcatonating proximity blocks\n")
        return self.cdf.concat(massive) #call the classmethod