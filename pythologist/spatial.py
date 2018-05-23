from multiprocessing import Pool
from functools import partial
from collections import OrderedDict
import pandas as pd
import numpy as np
import math, h5py
import pythologist
from plotnine import *
from scipy.stats import spearmanr
from scipy.spatial import cKDTree


def _access_tree(kd,sample,frame,markerA_name,markerB_name,k=1,dlim=None):
    results = []
    tree = kd[sample][frame][markerB_name]['kd']
    points2 = kd[sample][frame][markerB_name]['points']
    sub1 = kd[sample][frame][markerA_name]['sub']
    points1 = sub1.apply(lambda x: (x['x'],x['y'],x['id']),1).tolist()
    for point in points1:
        id1 = point[2]
        d,i = tree.query(point[0:2],k=k)
        if k == 1:
            d = [d]
            i = [i]
        for j,z in enumerate(i):
            m = OrderedDict({'sample':sample,'frame':frame,'markerA_name':markerA_name,'markerB_name':markerB_name,'markerA_id':id1,'markerB_id':points2[z],'distance':d[j],'rank':(j+1)})
            results.append(m)
    df = pd.DataFrame(results)
    if dlim is not None: df = df[df['distance']<=dlim]
    return df

def _get_trees(df,phenotypes,name):
        data = OrderedDict()
        dist = set()
        for phenotype in phenotypes:
            dist.add(phenotype[0])
            dist.add(phenotype[1])
        for phenotype in list(dist):
            sub = df[(df['full_phenotype']==phenotype)&(df['sample']==name[0])&(df['frame']==name[1])]
            points = sub.apply(lambda x: (x['x'],x['y'],x['id']),1)
            tree = cKDTree(list(zip(sub['x'].tolist(),sub['y'].tolist())))
            if phenotype not in data: 
                data[phenotype] = OrderedDict()
            data[phenotype]['kd'] = tree
            r = OrderedDict()
            data[phenotype]['points'] = [x[2] for x in points]
            data[phenotype]['sub'] = sub
        return data

def _run_through(kd,phenotypes,dlim,k,name):
    results = []
    for p1,p2 in phenotypes:
            r = _access_tree(kd,name[0],name[1],p1,p2,k=k,dlim=dlim)
            results.append(r)
    return pd.concat(results,sort=True)
    
def kNearestNeighborsCross(cf,phenotypes,k=1,threads=1,dlim=None):
    df = cf.df
    names = df.set_index(['sample','frame']).index.unique().tolist()
    pool = Pool(processes=threads)
    func = partial(_get_trees,df,phenotypes)
    v = pool.imap(func,names)
    pool.close()
    pool.join()
    kd = OrderedDict()
    for i,x in enumerate(v):
        name = names[i]
        if name[0] not in kd: kd[name[0]] = OrderedDict()
        kd[name[0]][name[1]] = x
    pool = Pool(processes=threads)
    func = partial(_run_through,kd,phenotypes,dlim,k)
    v = pool.imap(func,names)
    pool.close()
    pool.join()
    return pd.concat([x for x in v],sort=True)

#def _euclidian(pt1,pt2,dlim):
#    #dist = [(a - b)**2 for a, b in zip(pt1, pt2)]
#    #dist = math.sqrt(sum(dist))
#    return dist
def _pointcross(df,k,idlab,dlim,point):
    pt = (point['x'],point['y'])
    dists = df.set_index(idlab).apply(lambda x: math.sqrt((pt[0]-x['x'])**2+(pt[1]-x['y'])**2),1)
    if dlim is not None: dists = dists[dists <= dlim]
    dists = dists.nsmallest(k).sort_values()
    return OrderedDict(dists.to_dict())

def kNearestNeighborsCross2(df,
                      markerA,
                      markerB,
                      k=1,
                      xlab='x',
                      ylab='y',
                      classlab='full_phenotype',
                      idlab='id',
                      threads=1,
                      dlim = None):
    mA = df[df[classlab]==markerA][[xlab,ylab,idlab]]
    mB = df[df[classlab]==markerB][[xlab,ylab,idlab]]
    mList = mA.apply(lambda x: x.to_dict(),1).tolist()
    func = partial(_pointcross, mB, k, idlab,dlim)
    pool = Pool(processes=threads)
    v = pool.imap_unordered(func, mList)
    pool.close()
    pool.join()
    v = OrderedDict(zip([x[idlab] for x in mList], [x for x in v]))
    # lets move it back to our dataframe pardigm
    results = []
    for a in v.keys():
        for i,b in enumerate(v[a].keys()):
            results.append([a,b,i+1,v[a][b]])
    results = pd.DataFrame(results)
    results = results.rename(columns={0:'markerA_id',1:'markerB_id',2:'rank',3:'distance'})
    results['markerA_name'] = markerA
    results['markerB_name'] = markerB
    return results


class CellFrameNearestNeighbors:
    def __init__(self,cf,nn):
        self._cf = cf
        self._nn = nn
    def to_hdf(self,path):
        self._cf.to_hdf(path)
        self._nn.to_hdf(path,'nn',
                        mode='r+',
                        format='table',
                        complib='zlib',
                        complevel=9)
    @classmethod
    def read_hdf(cls,path):
       cf = pythologist.InFormCellFrame.read_hdf(path)
       nn = pd.read_hdf(path,'nn')
       v = cls(cf,nn)
       return v
    @property
    def cf(self): return self._cf
    @property
    def nn(self): return self._nn
    def plot_continuous_histogram(self,
                              markerA,
                              markerB,
                              continuous,
                              dlim=None,
                              step=30,
                              facets=['sample'],
                              standardize=True,
                              logscale=True,
                              minimum_A=0,
                              minimum_B=0,
                              rank=1,
                              scales=None):
        # The expression of markerA's continuous variable vs the nearest neighbor distance to markerB
        cf = self._cf
        nn = self._nn
        v = nn[(nn['markerA_name']==markerA)&(nn['markerB_name']==markerB)&(nn['rank']==rank)]
    
        if dlim is None: dlim = int(v['distance'].max())
        cnt = v[v['distance']<=dlim].groupby(['sample','frame']).count().\
            reset_index()[['sample','frame','rank']].rename(columns={'rank':'frame_count_A'})
        cntcut = cnt.copy()
        cntcut = cntcut[cntcut['frame_count_A']>=minimum_A]
        #if cntcut.shape[0] == 0: continue
        #if cut.shape[0] == 0: continue
        cnt2 = v[v['distance']<=dlim].groupby(['sample','frame']).\
            apply(lambda x: len(x['markerB_id'].unique().tolist())).\
            reset_index().rename(columns={0:'frame_count_B'})
        #if cnt2.shape[0] == 0: continue
        cntcut2 = cnt2.copy()
        cntcut2 = cntcut2[cntcut2['frame_count_B']>=minimum_B]
        #if cntcut2.shape[0] == 0: continue
        ocnt = cnt.merge(cnt2,on=['sample','frame'])
        ocnt['markerA'] = markerA
        ocnt['markerB'] = markerB
        ocnt['dlim'] = dlim

    
        v = v.merge(cntcut,on=['sample','frame']).reset_index().\
            merge(cntcut2,on=['sample','frame']).reset_index()
        choice = 'markerA_id'
        c = cf.df[facets+['id',continuous]]
        cuts = pd.cut(v['distance'],bins=range(0,dlim,step))
        v2 = v.copy()
        v2['bin'] = cuts
        v2 = v2.merge(c,left_on=facets+[choice],right_on=facets+['id'])
        v2 = v2.dropna()
        v2 = v2[v2['distance']<=dlim]
        v2 = v2[facets+['bin',continuous,'distance']]
        counts = v2.groupby(facets+['bin']).\
            apply(lambda x: len(list(x['distance']))).reset_index().\
            rename(columns={0:'count'})
        means = v2.groupby(facets+['bin']).\
            apply(lambda x: 
             sum(list(x[continuous]))/len(list(x[continuous]))).reset_index().\
            rename(columns={0:continuous})
        f = counts.merge(means,on=facets+['bin'])
        mymax = f.groupby(facets).max()[continuous].reset_index().rename(columns={continuous:'max'})
        mymin = f.groupby(facets).min()[continuous].reset_index().rename(columns={continuous:'min'})
        j = f.merge(mymax,on=facets).merge(mymin,on=facets)
        if standardize:
            n = j.apply(lambda x: (x[continuous]-x['min'])/(x['max']-x['min']),1)
            j[continuous] = n
        g = (ggplot(j.sort_values(facets),aes(x='bin',y='count',fill=continuous))
        + geom_bar(stat="identity")
        + theme_bw()
        + theme(axis_text_x=element_text(rotation=90, hjust=0))
        )
        if logscale:
            g += scale_y_log10()
        if scales:
            g += facet_wrap(facets,scales=scales)
        else:
            g += facet_wrap(facets)
        pair = v2.groupby(facets).apply(lambda x: {'continuous':x[continuous].tolist()
                                         ,'distance':x['distance'].tolist()})
        pair = pd.DataFrame(pair.apply(lambda x: dict(zip(('r','p'),
                                                  spearmanr(x['continuous'],x['distance'])))))
        pair = pair.apply(lambda x: pd.Series(x[0]),1).sort_values('r').reset_index()

        plot_data_frame = j
        cell_counts = ocnt
        correlations = pair
        return (g,plot_data_frame,cell_counts,correlations)