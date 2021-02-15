from multiprocessing import Pool
import json, sys, math
import pandas as pd
import numpy as np
from pythologist.selection import SubsetLogic as SL


def _get_frame_perm(myvars):
    cdf, random_state, phenotypes, i = myvars
    perm = cdf.permute_phenotype_labels(
        phenotypes,
        random_state = None if random_state is None else random_state+i
    ).contacts().frame_proportions()
    perm['perm'] = i
    return perm
def _get_sample_perm(myvars):
    cdf, random_state, phenotypes, i = myvars
    perm = cdf.permute_phenotype_labels(
        phenotypes,
        random_state = None if random_state is None else random_state+i
    ).contacts().sample_proportions()
    perm['perm'] = i
    return perm
def _analyze_perm(count,obs):
    low_break = len([x for x in obs if x <= count])/len(obs)
    high_break = len([x for x in obs if x >= count])/len(obs)
    mean = np.mean(obs)
    fold = math.log(count+1,2)-math.log(mean+1,)
    return pd.Series(dict(zip(
        ['count','n_perms','pvalue','fold'],
        (count,len(obs),min(low_break,high_break),fold)
    )))
def permute_frame_contacts(self,n_permutations=500,phenotypes=None,random_state=None,verbose=True,n_processes=1):
    if n_processes > 1 and random_state is None:
        raise ValueError("must provide a random state when multiprocessing")
    if phenotypes is None: phenotypes = self.phenotypes
    base_cdf = self.subset(SL(phenotypes=phenotypes))
    base = base_cdf.contacts().frame_proportions()
    perms = None
    if n_processes<=1:
        perms = pd.concat([_get_frame_perm((base_cdf,random_state,phenotypes,j)) for j in range(0,n_permutations)])
    else: 
        with Pool(processes=n_processes) as pool:
            perms = pd.concat([x for x in pool.imap_unordered(_get_frame_perm,[(base_cdf,random_state,phenotypes,j) for j in range(0,n_permutations)])])
    mergeon = ['project_id','project_name','sample_id','sample_name','frame_id','frame_name','region_label']
    _perms = perms.groupby(mergeon+['phenotype_label','neighbor_phenotype_label']).\
        apply(lambda x: [y for y in x['count']]).reset_index().\
            rename(columns={0:'obs'})
    _df = base.merge(_perms,on=mergeon+['phenotype_label','neighbor_phenotype_label'])
    _df = _df.set_index(mergeon+['phenotype_label','neighbor_phenotype_label','total','fraction']).\
        apply(lambda x: _analyze_perm(x['count'],x['obs']),1).reset_index()
    _df.loc[_df['total']==0,'n_perms'] = np.nan
    _df.loc[_df['total']==0,'pvalue'] = np.nan
    _df.loc[_df['total']==0,'fold'] = np.nan
    return _df

def permute_sample_contacts(self,n_permutations=500,phenotypes=None,random_state=None,verbose=True,n_processes=1):
    if n_processes > 1 and random_state is None:
        raise ValueError("must provide a random state when multiprocessing")
    if phenotypes is None: phenotypes = self.phenotypes
    base_cdf = self.subset(SL(phenotypes=phenotypes))
    base = base_cdf.contacts().sample_proportions()
    perms = None
    if n_processes<=1:
        perms = pd.concat([_get_sample_perm((base_cdf,random_state,phenotypes,j)) for j in range(0,n_permutations)])
    else: 
        with Pool(processes=n_processes) as pool:
            perms = pd.concat([x for x in pool.imap_unordered(_get_frame_perm,[(base_cdf,random_state,phenotypes,j) for j in range(0,n_permutations)])])
    mergeon = ['project_id','project_name','sample_id','sample_name','region_label']
    _perms = perms.groupby(mergeon+['phenotype_label','neighbor_phenotype_label']).\
        apply(lambda x: [y for y in x['count']]).reset_index().\
            rename(columns={0:'obs'})
    _df = base.merge(_perms,on=mergeon+['phenotype_label','neighbor_phenotype_label'])
    _df = _df.set_index(mergeon+['phenotype_label','neighbor_phenotype_label','total','fraction']).\
        apply(lambda x: _analyze_perm(x['count'],x['obs']),1).reset_index()
    _df.loc[_df['total']==0,'n_perms'] = np.nan
    _df.loc[_df['total']==0,'pvalue'] = np.nan
    _df.loc[_df['total']==0,'fold'] = np.nan
    return _df