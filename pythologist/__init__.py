import pandas as pd
import numpy as np
import sys, json, h5py
from pythologist.selection import SubsetLogic
from pythologist.measurements.counts import PercentageLogic
from pythologist.measurements.counts import Counts
from pythologist.measurements.spatial.contacts import Contacts
from pythologist.measurements.spatial.nearestneighbors import NearestNeighbors
from pythologist.measurements.spatial.cartesian import Cartesian
from pythologist.interface import SegmentationImages, phenotypes_to_regions
from pythologist.qc import QC

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
    _metadata = ['_microns_per_pixel','_db'] # for extending dataframe to include this property
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
        f.close()

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
        mergeon = ['project_id','project_name',
                'sample_id','sample_name',
                'frame_id','frame_name',
                ]
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
        #rows = rows.loc[rows['region_area_pixels']>0].copy()
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
            measured_regions (pandas.DataFrame): explicitly list the measured images and regions
            measured_phenotypes (list): explicitly list the phenotypes present

        Returns:
            NearestNeighbors: returns a class that holds nearest neighbor information for whatever phenotypes were in the CellDataFrame before execution.  This class is suitable for nearest neighbor and proximity operations.
        """
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
            CellProject: The new cell project
            CellDataFrame: The updated cell project
        """
        return phenotypes_to_regions(self,*args,**kwargs)

    def scored_to_phenotype(self,phenotypes):
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

def _extract_unique_keys_from_series(s):
    uni = pd.Series(s.apply(lambda x: json.dumps(x)).unique()).\
            apply(lambda x: json.loads(x)).apply(lambda x: set(sorted(x.keys())))
    return sorted(list(set().union(*list(uni))))
def _dict_rename(old,change):
    new_keys = [x if x not in change else change[x] for x in old.keys()]
    return dict(zip(new_keys, old.values()))




