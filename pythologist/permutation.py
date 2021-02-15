from multiprocessing import Pool
from tempfile import NamedTemporaryFile
import json, h5py, sys
import pandas as pd
def _func(b):
    return b.contacts().frame_proportions()

class Permutation:
    def __init__(self,h5_cache_path,mode='r+'):
        self.h5_cache_path=h5_cache_path
        self.mode = mode
        if self.mode != 'w':
            self.parameters = json.loads(str(h5py.File(self.h5_cache_path,'r')['parameters'].attrs['parameters']))
            self.cdf = CellDataFrame.read_hdf(h5_cache_path,'cdf')
        self._working_cdf = None
    def _get_working_cdf(self,cdf):
        if self.parameters['cell_proximity_cdfs']:
            nn = cdf.nearestneighbors(verbose=self.parameters['verbose'],
                                       per_phenotype_neighbors=self.parameters['k_neighbors'],
                                       include_self=True,
                                       min_neighbors=self.parameters['min_neighbors'],
                                       max_distance_px=None if 'max_distance_px' not in self.parameters else self.parameters['max_distance_px'],
                                       max_distance_um=None if 'max_distance_um' not in self.parameters else self.parameters['max_distance_um']
                                       )
            massive = []
            for i, (ref, mdf) in enumerate(nn.cell_proximity_cdfs(**self.parameters['cell_proximity_cdfs_parameters'])):
                if self.parameters['verbose'] and i%100==0:
                    sys.stderr.write("exploding the cell proxmity regions "+str(i)+"\r")
                mdf['channel_values'] = mdf['channel_values'].apply(lambda x: dict())
                mdf = mdf.rename_region(mdf.regions,str(ref['frame_id'])+'-'+str(ref.name))
                massive.append(mdf)
            massive = self.cdf.concat(massive)
            return massive 
        else:
            return cdf

    
    def set_parameters(self,**kwargs):
        """
        cdf
        phenotypes
        cell_proximity_cdfs
        cell_proximity_cdfs_parameters
        random_state
        n_permutations
        n_processes
        k_neighbors
        min_neighbors
        max_distance_px
        max_distance_um
        verbose
        """
        if self.mode != 'w': raise ValueError("These can only be established on setup")
        self.cdf = kwargs['cdf']
        self.cdf.to_hdf(self.h5_cache_path,'cdf',mode='w')
        _kwargs = kwargs.copy()
        del _kwargs['cdf']
        self.parameters = _kwargs
        var = json.dumps(self.parameters)
        with h5py.File(self.h5_cache_path,'r+') as h5:
            grp = h5.create_dataset("parameters",(1000, 1000), chunks=True)
            grp.attrs['parameters'] = var
    def _get_specific(self,_cdf,mp=True):
        if self.parameters['cell_proximity_cdfs'] and self.parameters['cell_proximity_cdfs_parameters'] is None:
            raise ValueError("You must set_cell_proximity_cdfs_parameters before running")
        if mp:
            with Pool(processes=self.parameters['n_processes']) as pool:
                fcnts = pd.concat([x for x in pool.imap_unordered(_func,_cdf.frame_region_generator())])
                return fcnts
        else:
            return pd.concat([_func(x) for x in _cdf.frame_region_generator()])
    def save_reference(self):
        fcnts = self._get_specific(self._get_working_cdf(self.cdf.copy()),mp=True)
        fcnts.to_hdf(self.h5_cache_path,'reference',mode='r+')
    def save_permutations(self,overwrite_all=False):
        if 'reference' not in [x for x in h5py.File(self.h5_cache_path,'r')]:
            raise ValueError("save the reference first")
        with Pool(processes=self.parameters['n_processes']) as pool:
            for i, x in pool.imap_unordered(_get_perm,[(j,self) for j in range(0,self.parameters['n_permutations'])]):
                x.to_hdf(self.h5_cach_path,'perm_'+str(i),mode='r+')
        #for i in range(0,self.parameters['n_permutations']):
def _get_perm(myvars):
    i, self = myvars
    if self.parameters['verbose']:
        sys.stderr.write("Calculating permutation "+str(i)+"\n")
    #if 'perm_'+str(i) in [x for x in h5py.File(self.h5_cache_path,'r')]:
    #    return
    return i, self._get_specific(
                                    self._get_working_cdf(
                                        self.cdf.permute_phenotype_labels(
                                            self.parameters['phenotypes'],
                                            random_state=None if self.parameters['random_state'] is None else self.parameters['random_state']+i
                                        )
                                    ),mp=False)
