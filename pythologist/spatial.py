from multiprocessing import Pool
from functools import partial
from collections import OrderedDict
import pandas as pd
import math, h5py
import pythologist
from plotnine import *
from scipy.stats import spearmanr


def _pointcross(df,k,xlab,ylab,idlab,point):
    pt = (point[xlab],point[ylab])
    dists = df.set_index(idlab).apply(lambda x: math.sqrt((pt[0]-x[xlab])**2+(pt[1]-x[ylab])**2),1).\
        nsmallest(k).sort_values()
    return OrderedDict(dists.to_dict())

def kNearestNeighborsCross(df,
                      markerA,
                      markerB,
                      k=1,
                      xlab='x',
                      ylab='y',
                      classlab='full_phenotype',
                      idlab='id',
                      threads=1):
    mA = df[df[classlab]==markerA][[xlab,ylab,idlab]]
    mB = df[df[classlab]==markerB][[xlab,ylab,idlab]]
    mList = mA.apply(lambda x: x.to_dict(),1).tolist()
    func = partial(_pointcross, mB, k, xlab, ylab, idlab)
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
                              rank=1):
        cf = self._cf
        nn = self._nn
        v = nn[(nn['markerA_name']==markerA)&(nn['markerB_name']==markerB)&(nn['rank']==rank)]
    
        if dlim is None: dlim = int(v['distance'].max())
        cnt = v[v['distance']<=dlim].groupby(['sample','frame']).count().\
            reset_index()[['sample','frame','rank']].rename(columns={'rank':'frame_count_A'})
        cntcut = cnt.copy()
        cntcut = cntcut[cntcut['frame_count_A']>=minimum_A]

        cnt2 = v[v['distance']<=dlim].groupby(['sample','frame']).\
            apply(lambda x: len(x['markerB_id'].unique().tolist())).\
            reset_index().rename(columns={0:'frame_count_B'})
        cntcut2 = cnt2.copy()
        cntcut2 = cntcut2[cntcut2['frame_count_B']>=minimum_B]
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
        + facet_wrap(facets)
        + theme_bw()
        )
        if logscale:
            g += scale_y_log10()
        
        pair = v2.groupby(facets).apply(lambda x: {'continuous':x['PDL1'].tolist()
                                         ,'distance':x['distance'].tolist()})
        pair = pd.DataFrame(pair.apply(lambda x: dict(zip(('r','p'),
                                                  spearmanr(x['continuous'],x['distance'])))))
        pair = pair.apply(lambda x: pd.Series(x[0]),1).sort_values('r').reset_index()


        return (g,j,ocnt,pair)