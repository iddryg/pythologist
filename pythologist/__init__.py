import pandas as pd
import numpy as np
import sys, json, h5py
from pythologist.selection import SubsetLogic
from pythologist.measurements.counts import PercentageLogic
from pythologist.measurements.counts import Counts
from pythologist.measurements.spatial.contacts import Contacts
from pythologist.measurements.spatial.nearestneighbors import NearestNeighbors
from pythologist.measurements.spatial.cartesian import Cartesian
from pythologist.interface import SegmentationImages, phenotypes_to_regions as interface_phenotypes_to_regions, fetch_single_segmentation_image_bytes, fetch_single_region_image_bytes
from pythologist.qc import QC
from itertools import chain
from pythologist._version import __version__


class CellDataSeries(pd.Series):
    @property
    def _constructor(self):
        return CellDataSeries
    @property
    def _constructor_expanddim(self):
        return CellDataFrame
    

class CellDataFrame(pd.DataFrame):
    """
    The **CellDataFrame** class is an extension of a pandas.DataFrame with
    per-cell rows that have region, binary calls, mutually exclusive phenotypes,
    cell locations, and cell-cell contact.

    Params: 
        microns_per_pixel (float): conversion factor that gets saved along with the dataframe once its set.  (20x vectra is a 0.496)
        db (CellProject): a storage class that has all the image and mask data
    """
    _metadata = ['_microns_per_pixel','_db','_version'] # for extending dataframe to include this property
    @property
    def _constructor(self):
        return CellDataFrame

    @property
    def _constructor_sliced(self):
        return CellDataSeries

    @property
    def _constructor_expanddim(self):
        return CellDataFrame

    def __init__(self,*args,**kw):
        kwcopy = kw.copy()
        super(CellDataFrame,self).__init__(*args,**kwcopy)
        self._version = __version__

    def copy(self,*args,**kw):
        """
        Create a copy. Do like a regular dataframe, but then also make sure to show copy all the individual entries with objects.
        """
        output = super(CellDataFrame,self).copy(*args,**kw)
        columns_with_objects = ['regions','scored_calls','phenotype_calls','channel_values','neighbors']
        for col_name in columns_with_objects:
            if col_name in self.columns:
                output[col_name] = [x if not isinstance(x,dict) else x.copy() for x in self[col_name]]
        output.microns_per_pixel = self.microns_per_pixel
        output._version = self.version
        output.db = self.db
        return output

    def get_valid_cell_indecies(self):
        """
        Return a dataframe of images present with 'valid' being a list of cell indecies that can be included
        """
        return pd.DataFrame(self).groupby(self.frame_columns).apply(lambda x: list(x['cell_index'])).\
            reset_index().rename(columns={0:'valid'})

    def prune_neighbors(self):
        """
        If the CellDataFrame has been subsetted, some of the cell-cell contacts may no longer be part of the the dataset.  This prunes those no-longer existant connections.

        Returns:
            CellDataFrame: A CellDataFrame with only valid cell-cell contacts
        """
        def _neighbor_check(neighbors,valid):
            if not neighbors==neighbors: return np.nan
            valid_keys = set(valid)&set(neighbors.keys())
            d = dict([(k,v) for k,v in neighbors.items() if k in valid_keys])
            return d
        fixed = self.copy()
        valid = self.get_valid_cell_indecies()
        valid = pd.DataFrame(self).merge(valid,on=self.frame_columns).set_index(self.frame_columns+['cell_index'])
        valid = valid.apply(lambda x: _neighbor_check(x['neighbors'],x['valid']),1).reset_index().\
            rename(columns={0:'new_neighbors'})
        fixed = fixed.merge(valid,on=self.frame_columns+['cell_index']).drop(columns='neighbors').\
            rename(columns={'new_neighbors':'neighbors'})
        fixed.microns_per_pixel = self.microns_per_pixel
        fixed.db = self.db
        #fixed.loc[:,'neighbors'] = list(new_neighbors)
        return fixed

    @property
    def frame_columns(self):
        """
        Returns a list of fields suitable for identifying the unique image frames
        """
        return ['project_id','project_name',
                'sample_id','sample_name',
                'frame_id','frame_name']
    @property
    def sample_columns(self):
        """
        Returns a list of fields suitable for identifying the unique samples
        """
        return ['project_id','project_name',
                'sample_id','sample_name']
    @property
    def project_columns(self):
        """
        Returns a list of fields suitable for identifying the unique projects
        """
        return ['project_id','project_name']

    def to_hdf(self,path,key,mode='a'):
        """
        Save the CellDataFrame to an hdf5 file.

        Args:
            path (str): the path to save to
            key (str): the name of the location to save it to
            mode (str): write mode
        """
        pd.DataFrame(self.serialize()).to_hdf(path,key,mode=mode,format='table',complib='zlib',complevel=9)
        f = h5py.File(path,'r+')
        f[key].attrs["microns_per_pixel"] = float(self.microns_per_pixel) if self.microns_per_pixel is not None else np.nan
        f[key].attrs["version"] = self.version
        f.close()
    def frame_region_generator(cdf):
        """
        Generator that produces individual regions of frames


        Returns:
            CellDataFrame
        """
        for project_id in cdf['project_id'].unique():
            project = cdf.loc[cdf['project_id']==project_id]
            for sample_id in project['sample_id'].unique():
                sample = project.loc[project['sample_id']==sample_id,:]
                for frame_id in sample['frame_id'].unique():
                    #print(frame_id)
                    frame = sample.loc[sample['frame_id']==frame_id,:]
                    for region_label in frame['region_label'].unique():
                        #print(region_label)
                        region = frame.loc[frame['region_label']==region_label].copy()
                        region.db = cdf.db
                        region.microns_per_pixel = cdf.microns_per_pixel
                        yield region
    def frame_generator(cdf):
        """
        Generator that produces individual frames


        Returns:
            CellDataFrame
        """
        for project_id in cdf['project_id'].unique():
            project = cdf.loc[cdf['project_id']==project_id]
            for sample_id in project['sample_id'].unique():
                sample = project.loc[project['sample_id']==sample_id,:]
                for frame_id in sample['frame_id'].unique():
                    #print(frame_id)
                    frame = sample.loc[sample['frame_id']==frame_id,:].copy()
                    frame.db = cdf.db
                    frame.microns_per_pixel = cdf.microns_per_pixel
                    yield frame
    def sample_generator(cdf):
        """
        Generator that produces individual frames


        Returns:
            CellDataFrame
        """
        for project_id in cdf['project_id'].unique():
            project = cdf.loc[cdf['project_id']==project_id]
            for sample_id in project['sample_id'].unique():
                sample = project.loc[project['sample_id']==sample_id,:].copy()
                sample.db = cdf.db
                sample.microns_per_pixel = cdf.microns_per_pixel
                yield sample
    def add_zeroed_phenotype(self,phenotype_label):
        """
        Add a phenotype to the mutually exclusive phenotypes, but it is set to zero. Raises an error if the phenotype already exists

        Args:
            phenotype_label (str): name of the phenotype to add

        Returns:
            CellDataFrame
        """
        def _add_item(x,phenotype_label):
            d = x.copy()
            d[phenotype_label] = 0
            return d
        if phenotype_label in self.phenotypes: raise ValueError("phenotype '"+str(phenotype_label)+"' already exists")
        cdf = self.copy()
        cdf['phenotype_calls'] = cdf['phenotype_calls'].apply(lambda x: _add_item(x,phenotype_label))
        return cdf


    def phenotypes_to_scored(self,phenotypes=None,overwrite=False):
        """
        Add mutually exclusive phenotypes to the scored calls

        Args:
            phenotypes (list): a list of phenotypes to add to scored calls.  if none or not set, add them all
            overwrite (bool): if True allow the overwrite of a phenotype, if False, the phenotype must not exist in the scored calls
        Returns:
            CellDataFrame
        """
        if not self.is_uniform(): raise ValueError("inconsistent phenotypes")
        if phenotypes is None: 
            phenotypes = self.phenotypes
        elif isinstance(phenotypes,str):
            phenotypes = [phenotypes]
        def _post(binary,phenotype_label,phenotypes,overwrite):
            d = binary.copy()
            if len(set(phenotypes)&set(list(binary.keys()))) > 0 and overwrite==False:
                raise ValueError("Error, phenotype already exists as a scored type")
            for label in phenotypes: d[label] = 0
            if phenotype_label == phenotype_label and phenotype_label in phenotypes:
                d[phenotype_label] = 1
            return d
        output = self.copy()
        output['scored_calls'] = output.apply(lambda x: 
                _post(x['scored_calls'],x['phenotype_label'],phenotypes,overwrite)
            ,1)
        return output


    @classmethod
    def concat(self,array_like):
        """
        Concatonate multiple CellDataFrames

        throws an error if the microns_per_pixel is not uniform across the frames

        Args:
            array_like (list): a list of CellDataFrames with 1 or more CellDataFrames

        Returns:
            CellDataFrame
        """
        arr = list(array_like)
        if len(set([x.microns_per_pixel for x in arr])) != 1:
            raise ValueError("Multiple microns per pixel set")
        cdf = CellDataFrame(pd.concat([pd.DataFrame(x) for x in arr]))
        cdf.microns_per_pixel = arr[0].microns_per_pixel
        return cdf


    @classmethod
    def read_hdf(cls,path,key=None):
        """
        Read a CellDataFrame from an hdf5 file.

        Args:
            path (str): the path to read from
            key (str): the name of the location to read from

        Returns:
            CellDataFrame
        """
        df = pd.read_hdf(path,key)
        df['scored_calls'] = df['scored_calls'].apply(lambda x: json.loads(x))
        df['channel_values'] = df['channel_values'].apply(lambda x: json.loads(x))
        df['regions'] = df['regions'].apply(lambda x: json.loads(x))
        df['phenotype_calls'] = df['phenotype_calls'].apply(lambda x: json.loads(x))
        df['neighbors'] = df['neighbors'].apply(lambda x: json.loads(x))
        df['neighbors'] = df['neighbors'].apply(lambda x:
                np.nan if not isinstance(x,dict) else dict(zip([int(y) for y in x.keys()],x.values()))
            )
        df['frame_shape'] = df['frame_shape'].apply(lambda x: tuple(json.loads(x)))
        df = cls(df)
        f = h5py.File(path,'r')
        mpp = f[key].attrs["microns_per_pixel"]
        if not np.isnan(mpp): df.microns_per_pixel = mpp
        _version  = None if 'version' not in f[key].attrs else f[key].attrs["version"]
        if not np.isnan(_version): df.version = _version
        f.close()
        return df

    def serialize(self):
        """
        Convert the data to one that can be saved in h5 structures

        Returns:
            pandas.DataFrame: like a cell data frame but serialized. columns
        """
        df = self.copy()
        df['scored_calls'] = df['scored_calls'].apply(lambda x: json.dumps(x))
        df['channel_values'] = df['channel_values'].apply(lambda x: json.dumps(x))
        df['regions'] = df['regions'].apply(lambda x: json.dumps(x))
        df['phenotype_calls'] = df['phenotype_calls'].apply(lambda x: json.dumps(x))
        df['neighbors'] = df['neighbors'].apply(lambda x: json.dumps(x))
        df['frame_shape'] = df['frame_shape'].apply(lambda x: json.dumps(x))
        return df

    @property
    def version(self):
        if not hasattr(self,'_version'): return None
        return self._version    

    @property
    def microns_per_pixel(self):
        """
        Read or store the micron's per pixel (float) value by reading or asigning to this
        """
        if not hasattr(self,'_microns_per_pixel'): return None
        return self._microns_per_pixel
    @microns_per_pixel.setter
    def microns_per_pixel(self,value):
        self._microns_per_pixel = value

    def is_uniform(self,verbose=True):
        """
        Check to make sure phenotype calls, or scored calls are consistent across all images / samples
        """
        uni = pd.Series(self['phenotype_calls'].apply(lambda x: json.dumps(x)).unique()).\
            apply(lambda x: json.loads(x)).apply(lambda x: tuple(sorted(x.keys()))).unique()
        if len(uni) > 1: 
            if verbose: sys.stderr.write("WARNING: phenotypes differ across the dataframe \n"+str(uni)+"\n")
            return False
        uni = pd.Series(self['scored_calls'].apply(lambda x: json.dumps(x)).unique()).\
            apply(lambda x: json.loads(x)).apply(lambda x: tuple(sorted(x.keys()))).unique()
        if len(uni) > 1: 
            if verbose: sys.stderr.write("WARNING: scored_calls differ across the dataframe \n"+str(uni)+"\n")
            return False
        return True

    @property
    def db(self):
        """
        Assign to this or read from this, the CellProject storage object
        """
        if not hasattr(self,'_db'): return None
        return self._db
    @db.setter
    def db(self,db):
        self._db = db

    @property
    def phenotypes(self):
        """
        Return the list of phenotypes present
        """
        return _extract_unique_keys_from_series(self['phenotype_calls'])

    @property
    def scored_names(self):
        """
        Return the list of binary feature names
        """
        return _extract_unique_keys_from_series(self['scored_calls'])

    @property
    def regions(self):
        """
        Return the list of region names
        """
        return _extract_unique_keys_from_series(self['regions'])

    def get_measured_regions(self):
        """
        Returns:
            pandas.DataFrame: Output a dataframe with regions and region sizes
        """
        mergeon = self.frame_columns
        temp = self.loc[:,mergeon+['regions']].\
            set_index(mergeon)['regions'].apply(json.dumps).\
            reset_index().drop_duplicates()
        temp['regions'] = temp['regions'].apply(json.loads)
        rows = []
        for i,r in temp.iterrows():
            for label in r['regions']:
                a = list(r.index)
                b = list(r.values)
                a = a+['region_label','region_area_pixels']
                b = b+[label,r['regions'][label]]
                rows.append(dict(zip(a,b)))
        rows = pd.DataFrame(rows).drop(columns='regions').\
            drop_duplicates()[mergeon+['region_label','region_area_pixels']]
        _cnt = self.groupby(self.frame_columns+['region_label']).count()[['cell_index']].reset_index().\
            rename(columns={'cell_index':'region_cell_count'})
        #rows = rows.loc[rows['region_area_pixels']>0].copy()
        rows = rows.merge(_cnt,on=self.frame_columns+['region_label'],how='left').fillna(0)
        rows['region_cell_count'] = rows['region_cell_count'].astype(int)
        return rows

    def segmentation_images(self,*args,**kwargs):
        """
        Use the segmented images to create per-image graphics

        Args:
            verbose (bool): output more details if true

        Returns:
            SegmentationImages: returns a class used to construct the image graphics
        """
        if not self.db: raise ValueError("Need to set db")
        segs = SegmentationImages.read_cellframe(self,*args,**kwargs)
        segs.microns_per_pixel = segs.microns_per_pixel
        return segs

    def nearestneighbors(self,*args,**kwargs):
        """
        Use the segmented images to create per-image graphics

        Args:
            verbose (bool): output more details if true
            per_phenotype_neighbors (int): number of neighbors of each phenotyhpe to find, default 50

        Returns:
            NearestNeighbors: returns a class that holds nearest neighbor information for whatever phenotypes were in the CellDataFrame before execution.  This class is suitable for nearest neighbor and proximity operations.
        """
        if 'per_phenotype_neighbors' not in kwargs: kwargs['per_phenotype_neighbors'] = 50
        if 'max_neighbors' not in kwargs: kwargs['max_neighbors'] = None
        n = NearestNeighbors.read_cellframe(self,*args,**kwargs)
        if 'measured_regions' in kwargs: n.measured_regions = kwargs['measured_regions']
        else: n.measured_regions = self.get_measured_regions()
        if 'measured_phenotypes' in kwargs: n.measured_phenotypes = kwargs['measured_phenotypes']
        else: n.measured_phenotypes = self.phenotypes
        n.microns_per_pixel = self.microns_per_pixel
        return n

    def contacts(self,*args,**kwargs):
        """
        Use assess the cell-to-cell contacts recorded in the celldataframe

        Returns:
            Contacts: returns a class that holds cell-to-cell contact information for whatever phenotypes were in the CellDataFrame before execution.  
        """
        n = Contacts.read_cellframe(self,prune_neighbors=True)
        if 'measured_regions' in kwargs: n.measured_regions = kwargs['measured_regions']
        else: n.measured_regions = self.get_measured_regions()
        if 'measured_phenotypes' in kwargs: n.measured_phenotypes = kwargs['measured_phenotypes']
        else: n.measured_phenotypes = self.phenotypes
        n.microns_per_pixel = self.microns_per_pixel
        return n

    def cartesian(self,subsets=None,step_pixels=100,max_distance_pixels=150,*args,**kwargs):
        """
        Return a class that can be used to create honeycomb plots

        Args:
            subsets (list): list of SubsetLogic objects
            step_pixels (int): distance between hexagons
            max_distance_pixels (int): the distance from each point by which to caclulate the quanitty of the phenotype for that area

        Returns:
            Cartesian: returns a class that holds the layout of the points to plot.
        """
        n = Cartesian.read_cellframe(self,subsets=subsets,step_pixels=step_pixels,max_distance_pixels=max_distance_pixels,prune_neighbors=False,*args,**kwargs)
        if 'measured_regions' in kwargs: n.measured_regions = kwargs['measured_regions']
        else: n.measured_regions = self.get_measured_regions()
        if 'measured_phenotypes' in kwargs: n.measured_phenotypes = kwargs['measured_phenotypes']
        else: n.measured_phenotypes = self.phenotypes
        n.microns_per_pixel = self.microns_per_pixel
        return n

    def counts(self,*args,**kwargs):
        """
        Return a class that can be used to access count densities

        Args:
            measured_regions (pandas.DataFrame): Dataframe of regions that are being measured (defaults to all the regions)
            measured_phenotypes (list): List of phenotypes present (defaults to all the phenotypes)
            minimum_region_size_pixels (int): Minimum region size to calculate counts on in pixels (Default: 1)
            minimum_denominator_count (int): Minimum denominator population count for percentage calculation (Default: 1)

        Returns:
            Counts: returns a class that holds the counts.
        """
        n = Counts.read_cellframe(self,prune_neighbors=False)
        if 'measured_regions' in kwargs: n.measured_regions = kwargs['measured_regions']
        else: n.measured_regions = self.get_measured_regions()
        if 'measured_phenotypes' in kwargs: n.measured_phenotypes = kwargs['measured_phenotypes']
        else: n.measured_phenotypes = self.phenotypes
        n.microns_per_pixel = self.microns_per_pixel
        if 'minimum_region_size_pixels' in kwargs: n.minimum_region_size_pixels = kwargs['minimum_region_size_pixels']
        else: n.minimum_region_size_pixels = 1
        if n.minimum_region_size_pixels < 1: raise ValueError("minimum_region_size_pixels must be at least 1")
        if 'minimum_denominator_count' in kwargs: n.minimum_denominator_count = kwargs['minimum_denominator_count']
        else: n.minimum_denominator_count = 1
        if n.minimum_denominator_count < 1: raise ValueError("minimum_denominator_count must be at least 1")
        return n

    def qc(self,*args,**kwargs):
        """
        Return a class that can be used to access QC reports

        Returns:
            QC: returns a class that can be used to interrogate the QC.
        """
        return QC(self,*args,**kwargs)

    def _shuffle_ids(self):
        together = []
        for frame_id in self['frame_id'].unique():
            v1 = self.loc[self['frame_id']==frame_id,['cell_index']].copy().reset_index(drop=True)
            v2 = v1.copy().sample(frac=1)
            v1['next_index'] = list(v2.index)
            v1['frame_id'] = frame_id
            together.append(v1)
        return pd.concat(together)

    ### Modifying functions
    def merge_scores(self,df_addition,reference_markers='all',
                                      addition_markers='all',on=['project_name','sample_name','frame_name','cell_index']):
        """
        Combine CellDataFrames that differ by score composition

        Args:
            df_addition (CellDataFrame): The CellDataFrame to merge scores in from
            reference_markers (list): which scored call names to keep in the this object (default: all)
            addition_markers (list): which scored call names to merge in (default: all)
            on (list): the features to merge cells on

        Returns:
            CellDataFrame,CellDataFrame: returns a passing CellDataFrame where merge criteria were met and a fail CellDataFrame where merge criteria were not met.
        """
        if isinstance(reference_markers, str):
            reference_markers = self.scored_names
        elif reference_markers is None: reference_markers = []
        if isinstance(addition_markers, str):
            addition_markers = df_addition.scored_names
        elif addition_markers is None: addition_markers = []

        df_addition = df_addition.copy()
        df_addition['_key'] = 1
        df = self.merge(df_addition[['scored_calls','_key']+on].rename(columns={'scored_calls':'_addition'}),
                            on = on,
                            how = 'left'
                        )

        df['_sub1'] = df['scored_calls'].apply(lambda x:
                dict((k,x[k]) for k in reference_markers)
            )
        df['_sub2'] = df['_addition'].apply(lambda x:
                dict({}) if x!=x else dict((k,x[k]) for k in addition_markers) # handle NaN where we fail to match properly treat as empty
            )
        # combine the two dictionaries
        df['scored_calls'] = df.apply(lambda x:
                {**x['_sub1'],**x['_sub2']}                    
            ,1)
        df = df.drop(columns=['_sub1','_sub2','_addition'])
        df = df.drop(columns='_key').copy(),df[df['_key'].isna()].drop(columns='_key').copy()
        if self.microns_per_pixel: df[0].microns_per_pixel = self.microns_per_pixel
        if self.microns_per_pixel: df[1].microns_per_pixel = self.microns_per_pixel
        return df

    def rename_scored_calls(self,change):
        """
        Change the names of scored call names, input dictionary change with {<current name>:<new name>} format, new name must not already exist

        Args:
            change (dict): a dictionary of current name keys and new name values

        Returns:
            CellDataFrame: The CellDataFrame modified.
        """
        output = self.copy()
        output['scored_calls'] = output.apply(lambda x:
          _dict_rename(x['scored_calls'],change)
          ,1)
        return output

    def zero_fill_missing_phenotypes(self):
        """
        Fill in missing phenotypes and scored types by listing any missing data as negative

        Returns:
            CellDataFrame: The CellDataFrame modified.
        """
        if self.is_uniform(verbose=False): return self.copy()
        output = self.copy()
        def _do_fill(d,names):
            old_names = list(d.keys())
            old_values = list(d.values())
            missing = set(names)-set(old_names)
            return dict(zip(old_names+list(missing),old_values+([0]*len(missing))))
        ## Need to make these uniform
        pnames = self.phenotypes
        output['phenotype_calls']= output.apply(lambda x:
            _do_fill(x['phenotype_calls'],pnames)
            ,1)
        return output
    
    def zero_fill_missing_scores(self):
        """
        Fill in missing phenotypes and scored types by listing any missing data as negative

        Returns:
            CellDataFrame: The CellDataFrame modified.
        """
        if self.is_uniform(verbose=False): return self.copy()
        output = self.copy()
        def _do_fill(d,names):
            old_names = list(d.keys())
            old_values = list(d.values())
            missing = set(names)-set(old_names)
            return dict(zip(old_names+list(missing),old_values+([0]*len(missing))))
        ## Need to make these uniform
        pnames = self.scored_names
        output['scored_calls']= output.apply(lambda x:
            _do_fill(x['scored_calls'],pnames)
            ,1)
        return output

    def filter_regions_by_area_pixels(self,region_area_pixels=1):
        """
        Strip away regions that dont have a minimum region area in square pixels
        """
        def _trim_regions_by_size(d,region_area_pixels):
            o = d.copy()
            for k,v in d.items():
                if v < region_area_pixels:
                    del o[k]
            return o
        output = self.copy()
        output['regions'] = output['regions'].\
            apply(lambda x: _trim_regions_by_size(x,region_area_pixels))
        output = output.loc[output.apply(lambda x: x['region_label'] in x['regions'],1),:]
        return output
    
    def drop_regions(self,region_names):
        """
        Take a region name or region names and rop those from the regions
        """
        def _trim_regions(d,region_names):
            o = d.copy()
            ks = set(o.keys())&set(region_names)
            for rn in ks:
                del o[rn]
            return o
        if isinstance(region_names,str):
            region_names = [region_names]
        output = self.copy()
        output = output.loc[~output['region_label'].isin(region_names),:]
        output['regions'] = output['regions'].\
            apply(lambda x: _trim_regions(x,region_names))
        return output

    def drop_scored_calls(self,names):
        """
        Take a name or list of scored call names and drop those from the scored calls

        Args:
            names (list): list of names to drop or a single string name to drop

        Returns:
            CellDataFrame: The CellDataFrame modified.
        """
        def _remove(calls,names):
            d = dict([(k,v) for k,v in calls.items() if k not in names])
            return d
        if isinstance(names, str):
            names = [names]
        output = self.copy()
        output['scored_calls'] = output['scored_calls'].\
            apply(lambda x: _remove(x,names))
        return output


    def subset(self,logic,update=False):
        """
        subset create a specific phenotype based on a logic, 
        logic is a 'SubsetLogic' class, 
        take union of all the phenotypes listed.  If none are listed use all phenotypes. 
        take the intersection of all the scored calls.

        Args:
            logic (SubsetLogic): A subsetlogic object to slice on
            update (bool): (default False) change the name of the phenotype according to the label in the subset logic

        Returns:
            CellDataFrame: The CellDataFrame modified.
        """
        pnames = self.phenotypes
        snames = self.scored_names
        data = self.copy()
        values = []
        phenotypes = logic.phenotypes
        if len(phenotypes)==0: phenotypes = pnames
        removing = set(self.phenotypes)-set(phenotypes)
        for k in phenotypes:
            if k not in pnames: raise ValueError("phenotype must exist in defined")
            temp = data.loc[data['phenotype_calls'].apply(lambda x: x[k]==1)].copy()
            if len(removing) > 0 and temp.shape[0] > 0:
                temp['phenotype_calls'] = temp.apply(lambda x:
                    dict([(k,v) for k,v in x['phenotype_calls'].items() if k not in removing])
                    ,1)
            values.append(temp)
        data = pd.concat(values)
        for k,v in logic.scored_calls.items():
            if k not in snames: raise ValueError("Scored name must exist in defined")
            myfilter = 0 if v == '-' else 1
            data = data.loc[data['scored_calls'].apply(lambda x: x[k]==myfilter)]
        data.microns_per_pixel = self.microns_per_pixel
        if update: 
            data['phenotype_calls'] = data['phenotype_calls'].apply(lambda x: {logic.label:1})
        data.fill_phenotype_label(inplace=True)
        data.db = self.db
        return data

    def threshold(self,phenotype,scored_name,positive_label=None,negative_label=None):
        """
        Split a phenotype on a scored_call and if no label is specified
        use the format '<phenotype> <scored_call><+/->'
        to specify a label give the positive and negative label

        Args:
            phenotype (str): name of the phenotype to threshold
            scored_name (str): scored call name to apply value from
            positive_label (str): name to apply for positive lable (default: <phenotype> <scored_call>+)
            negative_label (str): name to apply for negative lable (default: <phenotype> <scored_call>-)

        Returns:
            CellDataFrame: The CellDataFrame modified.
        """
        if positive_label is None and negative_label is not None or \
           negative_label is None and positive_label is not None: raise ValueError("Error if you want to specify labels, give both positive and negative")
        if phenotype not in self.phenotypes: raise ValueError("Error phenotype "+str(phenotype)+" is not in the data.")
        if scored_name not in self.scored_names: raise ValueError("Error scored_name "+str(scored_name)+" is not in the data.")
        if positive_label is None and negative_label is None:
            positive_label = phenotype+' '+scored_name+'+'
            negative_label = phenotype+' '+scored_name+'-'
        elif positive_label == negative_label: raise ValueError("Cant have the same label for positive and negative.")
        def _swap_in(d,pheno,scored,phenotype_calls,scored_calls,pos,neg):
            if pheno not in phenotype_calls.keys(): return d
            keepers = [(k,v) for k,v in phenotype_calls.items() if k!=phenotype]
            if scored not in scored_calls.keys(): raise ValueError("Error scored calls are not unified across samples")
            scored_value = scored_calls[scored]
            phenotype_value = phenotype_calls[pheno]
            if phenotype_value == 0:
                keepers += [(pos,0),(neg,0)]
            elif scored_value == 1:
                keepers += [(pos,1),(neg,0)]
            elif scored_value == 0:
                keepers += [(pos,0),(neg,1)]
            else: raise ValueError("Format error.  These values should only ever be zero or one.")
            return dict(keepers)
        data = self.copy()
        data['phenotype_calls'] = self.apply(lambda x:
                _swap_in(x,phenotype,scored_name,x['phenotype_calls'],x['scored_calls'],positive_label,negative_label)
            ,1)
        def _set_label(d):
            vals = [k for k,v in d.items() if v==1]
            return np.nan if len(vals) == 0 else vals[0]
        data['phenotype_label'] = data.apply(lambda x:
                _set_label(x['phenotype_calls'])
            ,1)
        return data.copy()

    def collapse_phenotypes(self,input_phenotype_labels,output_phenotype_label,verbose=True):
        """
        Rename one or more input phenotypes to a single output phenotype

        Args:
            input_phenotype_labels (list): A str name or list of names to combine
            output_phenotype_label (list): A str name to change the phenotype names to
            verbose (bool): output more details

        Returns:
            CellDataFrame: The CellDataFrame modified.
        """
        if isinstance(input_phenotype_labels,str): input_phenotype_labels = [input_phenotype_labels]
        bad_phenotypes = set(input_phenotype_labels)-set(self.phenotypes)
        if len(bad_phenotypes) > 0: raise ValueError("Error phenotype(s) "+str(bad_phenotypes)+" are not in the data.")
        data = self.copy()
        if len(input_phenotype_labels) == 0: return data
        def _swap_in(d,inputs,output):
            # Get the keys we need to merge together
            overlap = set(d.keys()).intersection(inputs)
            # if there are none to merge we're done already
            if len(overlap) == 0: return d
            keepers = [(k,v) for k,v in d.items() if k not in inputs]
            # combine anything thats not a keeper
            return dict(keepers+\
                        [(output_phenotype_label,max([d[x] for x in overlap]))])
        data['phenotype_calls'] = data.apply(lambda x:
            _swap_in(x['phenotype_calls'],input_phenotype_labels,output_phenotype_label)
            ,1)
        def _set_label(d):
            vals = [k for k,v in d.items() if v==1]
            return np.nan if len(vals) == 0 else vals[0]
        data['phenotype_label'] = data.apply(lambda x:
                _set_label(x['phenotype_calls']),1)
        return data

    def rename_phenotype(self,*args,**kwargs): 
        """simple alias for collapse phenotypes"""
        return self.collapse_phenotypes(*args,**kwargs)

    def combine_regions(self,input_region_labels,output_region_label,verbose=True):
        """
        Combine/rename one or more input regions to a single output region

        Args:
            input_region_labels (list): A str name or list of names to combine
            output_region_label (list): A str name to change the phenotype names to
            verbose (bool): output more details

        Returns:
            CellDataFrame: The CellDataFrame modified.
        """
        if isinstance(input_region_labels,str): input_region_labels = [input_region_labels]
        bad_regions = set(input_region_labels)-set(self.regions)
        if len(bad_regions) > 0: raise ValueError("Error regions(s) "+str(bad_regions)+" are not in the data.")
        data = self.copy()
        if len(input_region_labels) == 0: return data
        def _swap_in(d,inputs,output):
            # Get the keys we need to merge together
            overlap = set(d.keys()).intersection(inputs)
            # if there are none to merge we're done already
            if len(overlap) == 0: return d
            keepers = [(k,v) for k,v in d.items() if k not in inputs]
            # combine anything thats not a keeper
            return dict(keepers+\
                        [(output_region_label,sum([d[x] for x in overlap]))])
        data['regions'] = data.apply(lambda x:
            _swap_in(x['regions'],input_region_labels,output_region_label)
            ,1)
        data.loc[data['region_label'].isin(input_region_labels),'region_label'] = output_region_label
        return data
    def rename_region(self,*args,**kwargs): 
        """simple alias for combine phenotypes"""
        return self.combine_regions(*args,**kwargs)

    def fill_phenotype_label(self,inplace=False):
        """
        Set the phenotype_label column according to our rules for mutual exclusion
        """
        def _get_phenotype(d):
            vals = [k for k,v in d.items() if v ==  1]
            return np.nan if len(vals) == 0 else vals[0]
        if inplace:
            if self.shape[0] == 0: return self
            self['phenotype_label'] = self.apply(lambda x: _get_phenotype(x['phenotype_calls']),1)
            return
        fixed = self.copy()
        if fixed.shape[0] == 0: return fixed
        fixed['phenotype_label'] = fixed.apply(lambda x: _get_phenotype(x['phenotype_calls']),1)
        return fixed
    def fill_phenotype_calls(self,phenotypes=None,inplace=False):
        """
        Set the phenotype_calls according to the phenotype names
        """
        if phenotypes is None: phenotypes = list(self['phenotype_label'].unique())
        def _get_calls(label,phenos):
            d =  dict([(x,0) for x in phenos])
            if label!=label: return d # np.nan case
            d[label] = 1
            return d
        if inplace:
            self['phenotype_calls'] = self.apply(lambda x: _get_calls(x['phenotype_label'],phenotypes),1)
            return
        fixed = self.copy()
        fixed['phenotype_calls'] = fixed.apply(lambda x: _get_calls(x['phenotype_label'],phenotypes),1)
        return fixed
    def approximate_phenotypes_to_regions(self,allow_na_phenotype_label=False,na_phenotype_label='Unspecified'):
        """
        Create a new CellDataFrame where the region sizes are based 
        on the mutually exclusive phenotypes from the sum of cell areas.
        This does NOT create new region masks, and areas do not include space between cells.
        """
        mergeon=['project_id','project_name','sample_id','sample_name','frame_id','frame_name']
        region_names = self.phenotypes
        if allow_na_phenotype_label is False and self.loc[self['phenotype_label'].isna()].shape[0] > 0:
            raise ValueError("cannot set regions from phenotype labels unless allow_na_phenotype_label is True")
        elif allow_na_phenotype_label:
            region_names += [na_phenotype_label]
        df = pd.DataFrame(self.loc[:,mergeon+['phenotype_label','cell_area']])
        df.loc[df['phenotype_label'].isna(),'phenotype_label'] = na_phenotype_label
        df = df.groupby(mergeon+['phenotype_label']).sum()[['cell_area']].\
            rename(columns={'cell_area':'region_area_pixels'}).\
            reset_index()
        full = df.loc[:,mergeon].drop_duplicates()
        full['_key'] = 1
        regions = pd.DataFrame({'phenotype_label':region_names,'_key':[1 for x in range(0,len(region_names))]})
        df = full.merge(regions, on='_key').drop(columns=['_key']).merge(df,on=mergeon+['phenotype_label']).fillna(0)
        df = df.groupby(mergeon).apply(lambda x: dict(zip(x['phenotype_label'],x['region_area_pixels']))).\
            reset_index().rename(columns={0:'regions_new'})
        cdf2 = self.copy()
        cdf2 = cdf2.merge(df,on=mergeon)
        cdf2['region_label'] = cdf2['phenotype_label']
        cdf2.loc[cdf2['region_label'].isna(),'region_label'] = na_phenotype_label
        cdf2['regions'] = cdf2['regions_new']
        cdf2 = cdf2.drop(columns=['regions_new'])
        cdf2.microns_per_pixel = self.microns_per_pixel
        cdf2.db = self.db
        return cdf2

    def phenotypes_to_regions(self,*args,**kwargs):
        """
        Create a new Project where regions are replaced to be based on regions defined as phenotypes

        Args:
            path (str): Location to store a new hdf5 file containing a database update with new region images
            gaussian_sigma (float): the sigma parameter to the gaussian_filter function that says how much to 'blur'
            overwrite (bool): if True allows you to overwrite the path default (False)
            unset_label (str): A label to give regions that are unaccounted for
            project_name (str): the project name 

        Returns:
            CellDataFrame: The updated cell project
            CellProject: The new cell project
        """
        return interface_phenotypes_to_regions(self,*args,**kwargs)
    def fetch_single_segmentation_image_bytes(self,*args,**kwargs):
        """
        For a CellDataFrame sliced down to a single frame, get the image

        Args:
            schema (obj): schema defining the coloring
            background (tuple): integer tuple for background color

        Returns:
            bytes: A png image
        """
        return fetch_single_segmentation_image_bytes(self,*args,**kwargs)
    def fetch_single_region_image_bytes(self,*args,**kwargs):
        """
        For a CellDataFrame sliced down to a single frame, get the image

        Args:
            colors (obj): dict of hex colors keyed by region name
            background (str): hex for background color

        Returns:
            bytes: A png image
        """
        return fetch_single_region_image_bytes(self,*args,**kwargs)


    def regions_to_scored(self,regions=[]):
        """
        Covert the region calls to scored_calls

        Args: regions (list): a list of regions to use (default empty list will use all regions)
        """
        if len(regions) == 0: regions = self.regions
        if not isinstance(regions,list): raise ValueError("ERROR: regions is a list input")
        def _get_calls(current,region_label,regions):
            d = current.copy()
            for region in regions:
                if region in d.keys(): raise ValueError("ERROR: cannot overwrite a scored call.")
                d[region] = 0
                if region_label == region: d[region] =1
            return d
        fixed = self.copy()
        fixed['scored_calls'] = fixed.apply(lambda x: _get_calls(x['scored_calls'],x['region_label'],regions),1)
        return fixed


    def scored_to_phenotypes(self,phenotypes):
        """
        Convert binary pehnotypes to mutually exclusive phenotypes. 
        If none of the phenotypes are set, then phenotype_label becomes nan
        If any of the phenotypes are multiply set then it throws a fatal error.

        Args:
            phenotypes (list): a list of scored_names to convert to phenotypes

        Returns:
            CellDataFrame
        """
        def _apply_score(scored_calls,phenotypes):
            present = sorted(list(set(phenotypes)&set(scored_calls.keys())))
            total = sum([scored_calls[x] for x in present])
            if total > 1: 
                raise ValueError("You cant extract phenotypes from scores if they are not mutually exclusive")
            if total == 0: return np.nan
            for label in present:
                if scored_calls[label] == 1: return label
            raise ValueError("Should have hit an exit criteria already")
        output = self.copy()
        output['phenotype_label'] = output.apply(lambda x: _apply_score(x['scored_calls'],phenotypes),1)
        # now update the phenotypes with these
        output['phenotype_calls'] = output.apply(lambda x: 
            dict([(y,1 if x['phenotype_label']==y else 0) for y in phenotypes])
        ,1)
        return output
    def permute_phenotype_labels(self,phenotypes=None,
                                      channel_values=True,
                                      scored_calls=True,
                                      random_state=None):
        """
        Shuffle phenotype labels.  Defaults to shuffleling all labels within a frames regions.  Adjust this by modifying group_strategy.

        Args:
            phenotypes (list): a list of phenotype_labels to shuffle amongst eachother if None shuffle all
            channel_values (bool): include the channel_values in the shuffle
            scored_calls (bool): include the scored_calls in the shuffle
            random_state (int or numpy random state): pass to the pandas shuffle function

        Returns:
            CellDataFrame
        """
        phenotypes = phenotypes if phenotypes is not None else self.phenotypes 
        keepers = set(self.phenotypes)-set(phenotypes)
        if len(set(phenotypes)-set(self.phenotypes))>0:
            raise ValueError("Phenotypes must be defined.")
        keepers = self.loc[self['phenotype_label'].isin(keepers),:]
        permute = self.loc[self['phenotype_label'].isin(phenotypes),:]
        complete = []
        for i,block in enumerate(permute.frame_region_generator()):
            # we want to swap the labels of our phenotype_label
            # we also want to swap the scored calls if set to true
            # we also want to swap the channel_values if set to true
            mergeon = block.loc[:,['phenotype_label','phenotype_calls','scored_calls','channel_values']].\
                rename(columns={'phenotype_label':'plab',
                        'phenotype_calls':'pcall',
                        'scored_calls':'scall',
                        'channel_values':'cvalue'}).\
                reset_index(drop=True).\
                sample(frac=1,random_state=None if random_state is None else random_state+i).\
                reset_index(drop=True)
            block = block.reset_index(drop=True)
            block = block.merge(mergeon,left_index=True,right_index=True)
            block['phenotype_label'] = block['plab']
            block['phenotype_calls'] = block['pcall']
            if scored_calls:
                block['scored_calls'] = block['scall']
            if channel_values:
                block['channel_values'] = block['cvalue']
            block = block.drop(columns=['plab','pcall','scall','cvalue'])
            block.microns_per_pixel = self.microns_per_pixel
            block.db = self.microns_per_pixel
            complete.append(block)
        complete = self.__class__.concat(complete+[keepers]).\
            sort_values(['project_name','project_id','sample_name','sample_id','frame_name','frame_id','cell_index'])
        complete.db = self.db
        complete.microns_per_pixel = self.microns_per_pixel
        return complete.reset_index(drop=True)


    def threshold_on_mutually_exclusive_ordinal_labels(self,phenotype_label,ordinal_labels):
        """
        If mutually exclusive ordinal labels are present among the scoring, you can threshold a phenotype on these labels.

        Args:
            phenotype_label (str): a phenotype_label split based on the ordinal labels
            ordinal_labels (list): the list of ordinal labels to split the phenotype label on

        Returns:
            CellDataFrame
        """        
        def convert_labels(scored_calls,phenotype_calls,phenotype_label,ordinal_labels):
            fix = {}
            for k,v in phenotype_calls.items():
                if k != phenotype_label: fix[k] = v
            sanity_check = 0
            for ordinal_label in ordinal_labels:
                fix[phenotype_label+' '+ordinal_label] = \
                    1 if (phenotype_calls[phenotype_label]==1 and scored_calls[ordinal_label]==1) else 0
                sanity_check += scored_calls[ordinal_label]
            if sanity_check != 1: raise ValueError("ordinal labels not mutually exclusive.")
            return fix
        ndf = self.copy()
        print(ndf.shape)
        ndf['phenotype_calls'] = ndf.apply(lambda x: 
            convert_labels(x['scored_calls'],x['phenotype_calls'],phenotype_label,ordinal_labels)
        ,1)
        ndf = ndf.fill_phenotype_label()
        return ndf

    def convert_cascading_scores_to_mutually_exclusive_ordinal_binary(self,cascading_scored_calls,ordinal_labels):
        """
        If you have a cascade of scoring stored as binary calls, you can convert these to mutuallye exclusive binary calls for ordinal labels.

        Example is you have thresholds for 0/1, 1/2, and 2/3, you can convert these thresholds to 
        mutually exclusive +/- for 0,1,2,3

        Args:
            cascading_scored_calls (list): an ordered from lowest thresholds to greatest thresholds list of thresholds in scored_names
            ordinal_labels (list): the list of ordinal labels to split the phenotype label into

        Returns:
            CellDataFrame
        """   
        if len(ordinal_labels)-1!=len(cascading_scored_calls): 
            raise ValueError("You need one more ordinal label than the cascading thresholds")
        def do_conv(x,cascading_scored_calls,ordinal_labels):
            orig = x.copy()
            fix = {}
            for k,v in orig.items():
                if k not in cascading_scored_calls:
                    fix[k] = v
            ordinal_label = ordinal_labels[0]
            # initialize ordinal labels to zero
            for label in ordinal_labels:
                fix[label] = 1
            ordinal_label = ordinal_labels[0]
            for i,score_name in enumerate(cascading_scored_calls):
                # For each cascading score, see if there is something set to 1 thats greater
                # if there is, set the current ordinal to zero
                remaining = cascading_scored_calls[i:]
                #print([x[name] for name in remaining])
                v = sum([x[name] for name in remaining])
                if v == 0:
                    fix[ordinal_labels[i+1]]=0
                else:
                    fix[ordinal_labels[i]] = 0
            return fix
        ndf = self.copy()
        ndf['scored_calls'] = ndf['scored_calls'].\
            apply(lambda x: do_conv(x,cascading_scored_calls,ordinal_labels))
        return ndf

def _extract_unique_keys_from_series(s):
    v = [set(list(x.keys())) for x in s]
    return sorted(list(set(chain.from_iterable(v))))
    #uni = pd.Series(s.apply(lambda x: json.dumps(x)).unique()).\
    #        apply(lambda x: json.loads(x)).apply(lambda x: set(sorted(x.keys())))
    #return sorted(list(set().union(*list(uni))))
def _dict_rename(old,change):
    new_keys = [x if x not in change else change[x] for x in old.keys()]
    return dict(zip(new_keys, old.values()))




