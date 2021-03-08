import pandas as pd
import numpy as np
from itertools import chain
from pythologist.measurements import Measurement
from multiprocessing import Pool
import json, sys, math

def _find_one(d):
    for k in d.keys():
        if d[k] == 1: return k
    return np.nan

class Contacts(Measurement):
    @staticmethod
    def _preprocess_dataframe(cdf,*args,**kwargs):
        mergeon = ['project_id','project_name','sample_id','sample_name','frame_id','frame_name','region_label','cell_index']
        subset = cdf.loc[~cdf['neighbors'].isna()].copy().reset_index(drop=True)
        present = subset[mergeon].drop_duplicates()
        def _get_items(x):
            v = x['neighbors'].items()
            return ([x.name]*len(v),list(x['neighbors'].keys()),list(x['neighbors'].values()))
        data = subset.dropna(subset=['neighbors']).apply(_get_items,1) 
        myindex = list(chain(*[x[0] for x in data]))
        myneighbor = list(chain(*[x[1] for x in data]))
        myshared = list(chain(*[x[2] for x in data]))
        data = pd.DataFrame({'db_id':myindex,'neighbor_cell_index':myneighbor,'edge_shared_pixels':myshared})
        base = subset[['project_id','project_name',
                   'sample_id','sample_name',
                   'frame_id','frame_name',
                   'cell_index','region_label',
                   ]].merge(data,left_index=True,right_on='db_id').\
                   drop(columns='db_id')
        temp = cdf[mergeon +['edge_length','phenotype_calls']].copy()
        temp['phenotype_calls'] = temp['phenotype_calls'].apply(lambda x: _find_one(x))
        temp = temp.loc[~temp['phenotype_calls'].isna()].rename(columns={'phenotype_calls':'phenotype_label'})
        merged = temp.merge(base,on=mergeon)
        temp2 = temp.copy().rename(columns={'phenotype_label':'neighbor_phenotype_label',
                                            'edge_length':'neighbor_edge_length',
                                            'cell_index':'neighbor_cell_index'})
        merged = merged.merge(temp2,on=[x for x in mergeon if x!='cell_index'] + ['neighbor_cell_index'])
        return merged

    def frame_counts(self):
        mergeon = self.cdf.frame_columns+['region_label']
        mr = self.measured_regions.copy()
        mr['_key'] = 1
        cnts = self.groupby(mergeon+['phenotype_label','neighbor_phenotype_label']).\
            count()[['cell_index']].rename(columns={'cell_index':'contact_count'})
        cnts = cnts.reset_index()
        pheno1 = pd.DataFrame({'phenotype_label':self.measured_phenotypes})
        pheno1['_key'] = 1
        pheno2 = pheno1.copy().rename(columns={'phenotype_label':'neighbor_phenotype_label'})
        blank = mr.merge(pheno1,on='_key').merge(pheno2,on='_key').drop(columns='_key')
        cnts = blank.merge(cnts,on=mergeon+['phenotype_label','neighbor_phenotype_label'],how='left').fillna(0)
        cnts['region_area_mm2'] = cnts.apply(lambda x: 
            (x['region_area_pixels']/1000000)*(self.microns_per_pixel*self.microns_per_pixel),1)
        cnts['density_mm2'] = cnts.apply(lambda x: x['contact_count']/x['region_area_mm2'],1)
        cnts['contact_count'] = cnts['contact_count'].astype(int)
        _tot = cnts[mergeon+['phenotype_label','contact_count']].groupby(self.cdf.frame_columns+['region_label','phenotype_label']).sum().\
            rename(columns={'contact_count':'contact_total'})
        cnts = cnts.merge(_tot,on=self.cdf.frame_columns+['region_label','phenotype_label'])
        cnts['fraction'] = cnts.apply(lambda x: np.nan if x['contact_total']==0 else x['contact_count']/x['contact_total'],1)
        cnts['percent'] = cnts.apply(lambda x: np.nan if x['contact_total']==0 else 100*x['contact_count']/x['contact_total'],1)
        return cnts
    def sample_counts(self):
        mergeon = self.cdf.sample_columns+['region_label','phenotype_label','neighbor_phenotype_label']
        fc = self.measured_regions[self.cdf.frame_columns].\
            drop_duplicates().groupby(self.cdf.sample_columns).\
            count()[['frame_id']].rename(columns={'frame_id':'frame_count'})
        cnts = self.frame_counts().groupby(mergeon).\
            apply(lambda x:
                pd.Series(dict(zip(
                    ['cumulative_contact_count',
                     'cumulative_region_area_pixels',
                     'cumulative_region_area_mm2',
                     'cumulative_density_mm2',
                     'mean_density_mm2',
                     'stddev_density_mm2',
                     'stderr_density_mm2',
                     'measured_frame_count'
                    ],
                    [
                     x['contact_count'].sum(),
                     x['region_area_pixels'].sum(),
                     x['region_area_mm2'].sum(),
                     x['contact_count'].sum()/x['region_area_mm2'].sum(),
                     x['density_mm2'].mean(),
                     x['density_mm2'].std(),
                     x['density_mm2'].std()/np.sqrt(len(x['density_mm2'])),
                     len(x['density_mm2'])
                    ]
                )))
            ).reset_index()
        cnts = cnts.merge(fc,on=self.cdf.sample_columns)
        cnts['cumulative_contact_count'] = cnts['cumulative_contact_count'].astype(int)
        cnts['measured_frame_count'] = cnts['measured_frame_count'].astype(int)
        cnts['cumulative_region_area_pixels'] = cnts['cumulative_region_area_pixels'].astype(int)
        _tot = cnts[mergeon+['cumulative_contact_count']].groupby(self.cdf.sample_columns+['region_label','phenotype_label']).sum().\
            rename(columns={'cumulative_contact_count':'cumulative_contact_total'}).reset_index()
        cnts = cnts.merge(_tot,on=self.cdf.sample_columns+['region_label','phenotype_label'])
        cnts['cumulative_fraction'] = cnts.apply(lambda x: np.nan if x['cumulative_contact_total']==0 else x['cumulative_contact_count']/x['cumulative_contact_total'],1)
        cnts['cumulative_percent'] = cnts.apply(lambda x: np.nan if x['cumulative_contact_total']==0 else 100*x['cumulative_contact_count']/x['cumulative_contact_total'],1)
        return cnts
    def threshold(self,phenotype,contact_label=None):
        if contact_label is None: contact_label = phenotype+'/contact'
        def _add_score(d,value,label):
            d[label] = int(value)
            return d
        # for the given phenotype, define whether a cell is touching or not 
        cdf = self.cdf.copy()
        mergeon = cdf.frame_columns+['cell_index']
        contacts = self.loc[self['neighbor_phenotype_label']==phenotype,mergeon].drop_duplicates()
        contacts['_threshold'] = 1
        cdf = cdf.merge(contacts,on=mergeon,how='left')
        cdf.loc[cdf['_threshold'].isna(),'_threshold'] = 0
        cdf['scored_calls'] = cdf.apply(lambda x:
            _add_score(x['scored_calls'],x['_threshold'],contact_label)
        ,1)
        cdf.microns_per_pixel = self.microns_per_pixel
        cdf.db = self.cdf.db
        return cdf.drop(columns='_threshold')
    def permute(self,phenotypes=None,random_state=None):
        mergeon = ['project_name','project_id','sample_name','sample_id','frame_name','frame_id','region_label']
        phenotypes = self.cdf.phenotypes if phenotypes is None else phenotypes
        cdf2 = self.cdf.permute_phenotype_labels(phenotype_labels = phenotypes, random_state=random_state)
        data = pd.DataFrame(cdf2.loc[:,mergeon+['cell_index','phenotype_label']].rename(columns={'phenotype_label':'shuffled_phenotype_label'}))
        nn2 = self.copy().merge(data,on=mergeon+['cell_index']).\
            merge(data.rename(columns={'cell_index':'neighbor_cell_index'}),on=mergeon+['neighbor_cell_index'])
        nn2['phenotype_label'] = nn2['shuffled_phenotype_label_x']
        nn2['neighbor_phenotype_label'] = nn2['shuffled_phenotype_label_y']
        nn2 = nn2.drop(columns=['shuffled_phenotype_label_x','shuffled_phenotype_label_y'])
        nn2 = self.__class__(nn2)
        nn2.microns_per_pixel = self.microns_per_pixel
        nn2.cdf = cdf2
        nn2.verbose = self.verbose
        nn2.measured_phenotypes = self.measured_phenotypes
        nn2.measured_regions = self.measured_regions
        return nn2


    def permute_frame_counts(self,n_permutations=500,phenotypes=None,random_state=None,verbose=False,n_processes=1):
        if n_processes > 1 and random_state is None:
            raise ValueError("must provide a random state when multiprocessing")
        if phenotypes is None: phenotypes = self.cdf.phenotypes
        if verbose: sys.stderr.write("Calculating base contacts\n")
        base = self.frame_counts()
        if verbose: sys.stderr.write("Finished base contacts\n")
        perms = None
        if n_processes<=1:
            perms = pd.concat([_get_frame_perm((self,random_state,phenotypes,j,verbose)) for j in range(0,n_permutations)])
        else: 
            with Pool(processes=n_processes) as pool:
                perms = pd.concat([x for x in pool.imap_unordered(_get_frame_perm,[(self,random_state,phenotypes,j,verbose) for j in range(0,n_permutations)])])
        if verbose: sys.stderr.write("\n")
        mergeon = ['project_id','project_name','sample_id','sample_name','frame_id','frame_name','region_label']
        _perms = perms.groupby(mergeon+['phenotype_label','neighbor_phenotype_label']).\
            apply(lambda x: [y for y in x['contact_count']]).reset_index().\
                rename(columns={0:'perms'})
        _df = base.merge(_perms,on=mergeon+['phenotype_label','neighbor_phenotype_label'])
        _df = _df.set_index(mergeon+['phenotype_label',
                                     'neighbor_phenotype_label',
                                     'region_area_pixels',
                                     'region_area_mm2',
                                     'contact_total',
                                     'fraction',
                                     'percent',
                                     'density_mm2']).\
            apply(lambda x: _analyze_perm(x['contact_count'],x['perms']),1).reset_index()
        #_df.loc[_df['contact_total']==0,'n_perms'] = np.nan
        _df.loc[_df['contact_total']==0,'pvalue'] = np.nan
        _df.loc[_df['contact_total']==0,'fold'] = np.nan
        _df['perm_low_count'] = _df['perm_low_count'].astype(int)
        _df['perm_high_count'] = _df['perm_high_count'].astype(int)
        _df['n_perms'] = _df['n_perms'].astype(int)
        return _df

    def permute_sample_counts(self,n_permutations=500,phenotypes=None,random_state=None,verbose=False,n_processes=1):
        if n_processes > 1 and random_state is None:
            raise ValueError("must provide a random state when multiprocessing")
        if phenotypes is None: phenotypes = self.cdf.phenotypes
        if verbose: sys.stderr.write("Calculating base contacts\n")
        base = self.sample_counts()
        if verbose: sys.stderr.write("Finished base contacts\n")
        perms = None
        if n_processes<=1:
            perms = pd.concat([_get_sample_perm((self,random_state,phenotypes,j,verbose)) for j in range(0,n_permutations)])
        else: 
            with Pool(processes=n_processes) as pool:
                perms = pd.concat([x for x in pool.imap_unordered(_get_sample_perm,[(self,random_state,phenotypes,j,verbose) for j in range(0,n_permutations)])])
        mergeon = ['project_id','project_name','sample_id','sample_name','region_label']
        #print(perms.groupby(mergeon+['phenotype_label','neighbor_phenotype_label']).first().columns)
        _perms = perms.groupby(mergeon+['phenotype_label','neighbor_phenotype_label']).\
            apply(lambda x: [y for y in x['cumulative_contact_count']]).reset_index().\
                rename(columns={0:'perms'})
        _df = base.merge(_perms,on=mergeon+['phenotype_label','neighbor_phenotype_label'])
        _df = _df.set_index(mergeon+['phenotype_label',
                                     'neighbor_phenotype_label',
                                     'cumulative_region_area_pixels',
                                     'cumulative_region_area_mm2',
                                     'cumulative_contact_total',
                                     'cumulative_density_mm2',
                                     'mean_density_mm2',
                                     'stddev_density_mm2',
                                     'stderr_density_mm2',
                                     'measured_frame_count',
                                     'frame_count',
                                     'cumulative_fraction',
                                     'cumulative_percent']).\
            apply(lambda x: _analyze_perm(x['cumulative_contact_count'],x['perms']),1).reset_index()
        #_df.loc[_df['cumulative_contact_total']==0,'n_perms'] = np.nan
        _df.loc[_df['cumulative_contact_total']==0,'pvalue'] = np.nan
        _df.loc[_df['cumulative_contact_total']==0,'fold'] = np.nan
        _df['perm_low_count'] = _df['perm_low_count'].astype(int)
        _df['perm_high_count'] = _df['perm_high_count'].astype(int)
        _df['n_perms'] = _df['n_perms'].astype(int)
        _df = _df.rename(columns={'contact_count':'cumulative_contact_count'})
        _df['cumulative_contact_count'] = _df['cumulative_contact_count'].astype(int)
        return _df


def _get_frame_perm(myvars):
    contacts, random_state, phenotypes, i, verbose = myvars
    perm = contacts.permute(
        phenotypes= phenotypes,
        random_state = None if random_state is None else random_state+i
    ).frame_counts()
    perm['perm'] = i
    if verbose: sys.stderr.write("Finished perm "+str(i)+"\r")
    return perm
def _get_sample_perm(myvars):
    contacts, random_state, phenotypes, i, verbose = myvars
    perm = contacts.permute(
        phenotypes=phenotypes,
        random_state = None if random_state is None else random_state+i
    ).sample_counts()
    perm['perm'] = i
    if verbose: sys.stderr.write("Finished perm "+str(i)+"\r")
    return perm
def _analyze_perm(count,perms):
    perm_low_count = len([x for x in perms if x <= count])
    low_break = perm_low_count/len(perms)
    perm_high_count = len([x for x in perms if x >= count])
    high_break = perm_high_count/len(perms)
    mean = np.mean(perms)
    fold = math.log(count+1,2)-math.log(mean+1,2)
    return pd.Series(dict(zip(
        ['contact_count','n_perms','perm_low_count','perm_high_count','pvalue','fold'],
        (count,len(perms),perm_low_count,perm_high_count,min(low_break,high_break),fold)
    )))