import pandas as pd
def _base_neighbors(cdf,cache=True):
    mergeon = ['project_id','sample_id','frame_id','cell_index']
    subset = cdf[~cdf['neighbors'].isna()].copy()
    present = subset[mergeon].drop_duplicates()
    if cache and cdf._neighbors is not None: 
        return cdf._neighbors.copy().merge(present,on=mergeon)
    data = subset.apply(lambda x: 
        pd.DataFrame({
            'project_id':x['project_id'],
            'project_name':x['project_name'],
            'sample_id':x['sample_id'],
            'sample_name':x['sample_name'],
            'frame_id':x['frame_id'],
            'frame_name':x['frame_name'],
            'cell_index':x['cell_index'],
            'region_label':x['region_label'],
            'neighbor_cell_index':list(x['neighbors'].keys()),
            'edge_shared_pixels':list(x['neighbors'].values())
        })
    ,1)
    data = pd.concat(data.tolist())
    data = data.merge(present,on=mergeon)
    data['neighbor_cell_index'] = data['neighbor_cell_index'].astype(int)
    data['edge_shared_pixels'] = data['edge_shared_pixels'].astype(int)
    cdf._neighbors = data.copy()
    return data

def neighbors(cdf,cache=True):
    base = _base_neighbors(cdf,cache=cache)
    def _find_one(d):
        for k in d.keys():
            if d[k] == 1: return k
        return np.nan
    temp = cdf[['frame_id','cell_index','edge_length','phenotype_calls']].copy()
    temp['phenotype_calls'] = temp['phenotype_calls'].apply(lambda x: _find_one(x))
    temp = temp.loc[~temp['phenotype_calls'].isna()].rename(columns={'phenotype_calls':'phenotype'})
    merged = temp.merge(base,on=['frame_id','cell_index'])
    temp2 = temp.copy().rename(columns={'phenotype':'neighbor_phenotype','edge_length':'neighbor_edge_length','cell_index':'neighbor_cell_index'})
    merged = merged.merge(temp2,on=['frame_id','neighbor_cell_index'])
    return merged