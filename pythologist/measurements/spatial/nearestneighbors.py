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
        if 'min_neighbors' not in kwargs: raise ValueError('max_neighbors must be defined')
        k_neighbors = kwargs['min_neighbors']
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
        nn['min_neighbors'] = k_neighbors
        return nn




        #step_pixels = kwargs['step_pixels']
        #max_distance_pixels = kwargs['max_distance_pixels']
        def _mindist_nodiag(pts1,pts2):
            mat = cdist(list(pts1),list(pts2))
            if len(pts1)==len(pts2) and set(pts1.index) == set(pts2.index): 
                np.fill_diagonal(mat,np.nan)
            dmat = pd.DataFrame(mat)
            # remove if they are all nan
            worst = pd.DataFrame(mat).isna().all(1)
            dmat.loc[worst]=999999999
            mat = np.array(dmat)
            #print(mat)
            matmin = np.nanargmin(mat,axis=1).astype(float)
            matmin[worst] = np.nan
            #print(matmin)
            data = [[pts1.index[i],pts1.iloc[i],np.nan,np.nan,np.nan] if np.isnan(y) else
                    (pts1.index[i],pts1.iloc[i],pts2.index[int(y)],pts2.iloc[int(y)],mat[i,int(y)]) \
                        for i,y in enumerate(matmin)]
            data = pd.DataFrame(data,columns=['cell_index','cell_coord','neighbor_cell_index','neighbor_cell_coord','neighbor_distance_px'])
            return data.dropna()
        def _combine_dfs(minima,index,index_names):
            n1 = minima
            n1['_key'] = 1
            n2 = pd.DataFrame(index,index=index_names).T
            n2['_key'] = 1
            return n2.merge(n1,on='_key').drop(columns='_key')
        cdf = cdf.copy()
        if 'verbose'  in kwargs and kwargs['verbose']: sys.stderr.write("read phenotype label\n")
        mr = cdf.get_measured_regions().drop(columns='region_area_pixels')
        #cdf['phenotype_label'] = cdf.apply(lambda x: 
        #        [k for k,v in x['phenotype_calls'].items() if v==1]
        #    ,1).apply(lambda x: np.nan if len(x)==0 else x[0])
        cdf['phenotype_label'] = cdf['phenotype_calls'].\
            apply(lambda x: dict((v,k) for k, v in x.items())).\
            apply(lambda x: np.nan if 1 not in x else x[1])
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
        if k_neighbors > self.iloc[0]['min_neighbors']:
            raise ValueError("must select a k_neighbors smaller or equal to the min_neighbors used to generate the NearestNeighbors object")
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

        output = self.cdf.copy()
        mergeon = output.frame_columns+['region_label','cell_index']
        cdf = output.merge(nn1[mergeon+['_threshold']],on=mergeon)
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

