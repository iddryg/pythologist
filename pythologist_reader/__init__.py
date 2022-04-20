import pandas as pd
import numpy as np
import h5py, os, json, sys, shutil
from uuid import uuid4
from pythologist_image_utilities import map_image_ids
from pythologist_reader.qc import QC
from pythologist import CellDataFrame
from tempfile import SpooledTemporaryFile

"""
These are classes to help deal with cell-level image data
"""

class FrameMaskLabel(object):
    """
    A genetic class for holding a generic labled image mask
    """
    def __init__(self,mask_image,region_group,labels):
        self._mask_image = mask_image
        self._mask_image_region_group = region_group
        self._mask_iamge_label_dictionary = labels
        if region_group is not isinstance(region_group,str):
            raise ValueError("Must provide a string label for region_group")

    @property
    def region_group(self):
        return self._mask_image_region_group
    @property
    def labels(self):
        return self._mask_image_label_dictionary

class CellFrameGeneric(object):
    """
    A generic CellFrameData object
    """
    def __init__(self):
        self.verbose = False
        self._processed_image_id = None
        self._images = {}                      # Database of Images
        self._id = uuid4().hex
        self.frame_name = None
        self.data_tables = {
        'cells':{'index':'cell_index',            
                  'columns':['x','y','cell_area','edge_length']},
        # Tables for setting measurements
        'cell_measurements':{'index':'measurement_index', 
                             'columns':['cell_index','statistic_index','measurement_feature_index','channel_index','value']},
        'measurement_features':{'index':'measurement_feature_index',
                                'columns':['measurement_feature_label']},
        'measurement_channels':{'index':'channel_index',
                                'columns':['channel_label','channel_abbreviation','image_id']},
        'measurement_statistics':{'index':'statistic_index',
                                  'columns':['statistic_label']},
        'segmentation_images':{'index':'db_id',
                 'columns':['segmentation_label','image_id']},
        'cell_interactions':{'index':'db_id', 
                             'columns':['cell_index','neighbor_cell_index','pixel_count','touch_distance']
        },
        # Tables for setting regions
        'cell_regions':{
            'index':'db_id',
            'columns':['cell_index','region_index']
        },
        'regions':{
            'index':'region_index',
            'columns':['region_group_index','region_label','region_size','image_id']
        },
        'region_groups':{
            'index':'region_group_index',
            'columns':['region_group','region_group_description']
        },
        # Tables for setting features
        'cell_features':{'index':'db_id',
                         'columns':['cell_index','feature_definition_index']
                        },
        'feature_definitions':{'index':'feature_definition_index',
                                'columns':['feature_index','feature_value_label','feature_value']},
        'features':{
            'index':'feature_index',
            'columns':['feature_label','feature_description']
        },
        # Supplemental
        'mask_images':{
            'index':'db_id',
            'columns':['mask_label','image_id']
        },
        #'thresholds':{
        #    'index':'gate_index',
        #    'columns':['threshold_value','statistic_index',
        #               'measurement_feature_index','channel_index',
        #               'region_index']
        #},
        'cell_tags':{'index':'db_id',            
                     'columns':['tag_index','cell_index']},
        'tags':{'index':'tag_index',
                'columns':['tag_label']}
                           }
        self._data = {} # Do not acces directly. Use set_data_table and get_data_table to access.
        for x in self.data_tables.keys(): 
            self._data[x] = pd.DataFrame(columns=self.data_tables[x]['columns'])
            self._data[x].index.name = self.data_tables[x]['index']
    @property
    def id(self):
        """
        Returns the project UUID4
        """
        return self._id

    @property
    def shape(self):
        """
        Returns the (tuple) shape of the image (rows,columns)
        """
        return self.processed_image.shape
    
    @property
    def processed_image_id(self):
        """
        Returns (str) id of the frame object
        """
        return self._processed_image_id
    @property
    def processed_image(self):
        """
        Returns (numpy.array) of the processed_image
        """
        return self._images[self._processed_image_id].copy()
    def set_processed_image_id(self,image_id):
        """
        Args:
            image_id (str): set the id of the frame object
        """
        self._processed_image_id = image_id

    @property
    def table_names(self):
        """
        Return a list of data table names
        """
        return list(self.data_tables.keys())

    def import_cell_features(self,cell_frame,feature_names,name_change_table={}):
        """
        Import cell features from one cell_frame into another
        """
        

        def _reassemble_feature_table(cell_frame):
            _features = cell_frame.get_data('features')
            _feature_definitions = cell_frame.get_data('feature_definitions')
            _cell_features = cell_frame.get_data('cell_features')
            return _features.merge(_feature_definitions,left_index=True,right_on='feature_index',how='left').\
                merge(_cell_features,left_index=True,right_on='feature_definition_index',how='left')

        _f1 = _reassemble_feature_table(self).\
             drop(columns = ['feature_index','feature_definition_index'])
        _f2 = _reassemble_feature_table(cell_frame).\
            drop(columns = ['feature_index','feature_definition_index'])
        _f2['feature_label'] = _f2['feature_label'].apply(lambda x: x if x not in name_change_table else name_change_table[x])

        # filter to feature names we are interested in
        _f2 = _f2.loc[_f2['feature_label'].isin(feature_names)]

        # make a new massive table
        _comb = pd.concat([_f1,_f2]).sort_values(['feature_label','feature_value']).reset_index(drop=True)

        # Now reindex the feature table
        _features = _comb.loc[:,['feature_label','feature_description']].drop_duplicates().\
            reset_index(drop=True)
        _features.index.name = 'feature_index'

        # Now reindex feature definitions
        _feature_definitions = _comb.loc[:,['feature_value_label',
                                    'feature_value',
                                    'feature_label']].drop_duplicates().\
            merge(_features[['feature_label']].reset_index(),on=['feature_label']).\
            drop(columns=['feature_label']).reset_index(drop=True)
        _feature_definitions.index.name = 'feature_definition_index'

        _cell_features = _comb.copy().dropna(subset=['cell_index'])
        _cell_features['cell_index'] = _cell_features['cell_index'].astype(int)
        _cell_features = _cell_features.\
            merge(_features[['feature_label']].reset_index(),on=['feature_label']).\
            drop(columns=['feature_label']).\
            merge(_feature_definitions[['feature_value','feature_index','feature_value_label']].\
                  reset_index(),
                  on = ['feature_value','feature_value_label','feature_index']
                 ).\
            drop(columns=['feature_value','feature_value_label','feature_index','feature_description']).\
            sort_values(['cell_index','feature_definition_index']).\
            reset_index(drop=True)
        _cell_features.index.name = 'db_id'
        output = self.copy()
        output.set_data('features',_features)
        output.set_data('feature_definitions',_feature_definitions)
        output.set_data('cell_features',_cell_features)
        return output


    def set_data(self,table_name,table):
        """
        Set the data table

        Args:
            table_name (str): the table name 
            table (pd.DataFrame): the input table
        """
        # Assign data to the standard tables. Do some column name checking to make sure we are getting what we expect
        if table_name not in self.data_tables: raise ValueError("Error table name doesn't exist in defined formats")
        if set(list(table.columns)) != set(self.data_tables[table_name]['columns']): raise ValueError("Error column names don't match defined format\n"+\
                                                                                            str(list(table.columns))+"\n"+\
                                                                                            str(self.data_tables[table_name]['columns']))
        if table.index.name != self.data_tables[table_name]['index']: raise ValueError("Error index name doesn't match defined format")
        self._data[table_name] = table.loc[:,self.data_tables[table_name]['columns']].copy() # Auto-sort, and assign a copy so we aren't ever assigning by reference

    def set_regions(self,region_group_name,regions,description="",use_processed_region=True,unset_label='undefined',verbose=False):
        """
        Alter the regions in the frame

        Args:
            regions (dict): a dictionary of mutually exclusive region labels and binary masks
                            if a region does not cover all the workable areas then it will be the only label
                            and the unused area will get the 'unset_label' as a different region
            use_processed_region (bool): default True keep the processed region subtracted
            unset_label (str): name of unset regions default (undefined)
        """
        
        ### delete our current regions
        ##raise ValueError("Function not brought to compatibility")
        ##regions = regions.copy()
        ##image_ids = list(self.get_data('mask_images')['image_id'])
        ##image_ids = [x for x in image_ids if x != self.processed_image_id]
        ##for image_id in image_ids: del self._images[image_id]

        labels = list(regions.keys())
        ids = [uuid4().hex for x in labels]
        sizes = [regions[x].sum() for x in labels]
        remainder = np.ones(self.processed_image.shape)
        if use_processed_region: remainder = self.processed_image

        for i,label in enumerate(labels):
            my_image = regions[label]
            if use_processed_region: my_image = my_image&self.processed_image
            self._images[ids[i]] = my_image
            remainder = remainder & (~my_image)

        if verbose: sys.stderr.write("Remaining areas after setting are "+str(remainder.sum().sum())+"\n")

        if remainder.sum().sum() > 0:
            labels += [unset_label]
            sizes += [remainder.sum().sum()]
            ids += [uuid4().hex]
            self._images[ids[-1]] = remainder
            regions[unset_label] = remainder

        # add a new region group
        _rgdf = self.get_data('region_groups')
        _region_group_index = _rgdf.index.max()+1
        _rgnew = pd.Series({'region_group':region_group_name,'region_group_description':description},name=_region_group_index)
        _rgdf = _rgdf.append(_rgnew)
        self.set_data('region_groups',_rgdf)

        _region_index_start = self.get_data('regions').index.max()+1
        regions2 = pd.DataFrame({'region_group_index':[_region_group_index for x in labels],
                                 'region_label':labels,
                                 'region_size':sizes,
                                 'image_id':ids
                                },index=[x for x in range(_region_index_start,_region_index_start+len(labels))])
        regions2.index.name = 'region_index'
        #print(regions2)
        _regcomb = self.get_data('regions').append(regions2)
        self.set_data('regions',_regcomb)
        def get_label(x,y,regions_dict):
            for label in regions_dict:
                if regions_dict[label][y][x] == 1: return label
            return np.nan
            raise ValueError("Coordinate is out of bounds for all regions.")
        recode = self.get_data('cells')[['x','y']].copy()
        recode['region_index'] = 0
        recode['new_region_label'] = recode.apply(lambda x: get_label(x['x'],x['y'],regions),1)
        ## see how many we need to drop because the centroid fall in an unprocessed region
        if verbose: sys.stderr.write(str(recode.loc[recode['new_region_label'].isna()].shape[0])+" cells with centroids beyond the processed region are being dropped\n")
        recode = recode.loc[~recode['new_region_label'].isna()].copy()
        recode = recode.drop(columns='region_index').reset_index().\
            merge(regions2[['region_label']].reset_index(),
                  left_on='new_region_label',right_on='region_label').\
            drop(columns=['region_label','new_region_label']).set_index('cell_index').\
            reset_index().drop(columns=['x','y'])
        recode.index.name = 'db_id'
        # now add the recoded to the current cell_regions
        recode_comb = self.get_data('cell_regions').append(recode)
        self.set_data('cell_regions',recode_comb)
        return


    def get_data(self,table_name): 
        """
        Get the data table

        Args:
            table_name (pandas.DataFrame): the table you access by name
        """
        return self._data[table_name].copy()

    def read_hdf(self,h5file,location=''):
        if location != '': location = location.split('/')
        else: location = []
        f = h5py.File(h5file,'r')
        subgroup = f
        for x in location:
            subgroup = subgroup[x]
        table_names = [x for x in subgroup['data']]
        for table_name in table_names:
            loc = '/'.join(location+['data',table_name])
            #print(loc)
            self.set_data(table_name,pd.read_hdf(h5file,loc))
        # now get images
        image_names = [x for x in subgroup['images']]
        for image_name in image_names:
            self._images[image_name] = np.array(subgroup['images'][image_name])
        self.frame_name = subgroup['meta'].attrs['frame_name']
        self._id = subgroup['meta'].attrs['id']
        self.set_processed_image_id(subgroup['meta'].attrs['processed_image_id'])
        return

    def to_hdf(self,h5file,location='',mode='w'):
        f = h5py.File(h5file,mode)
        f.create_group(location+'/data')
        f.create_group(location+'/images')
        #f.create_group(location+'/meta')
        f.close()
        for table_name in self.data_tables.keys():
            data_table = self.get_data(table_name)
            data_table.to_hdf(h5file,
                              location+'/data/'+table_name,
                              mode='a',
                              format='table',
                              complib='zlib',
                              complevel=9)
        f = h5py.File(h5file,'a')
        for image_id in self._images.keys():
            f.create_dataset(location+'/images/'+image_id,data=self._images[image_id],compression='gzip',compression_opts=9)
        dset = f.create_dataset(location+'/meta', (100,), dtype=h5py.special_dtype(vlen=str))
        dset.attrs['frame_name'] = self.frame_name
        dset.attrs['processed_image_id'] = self.processed_image_id
        dset.attrs['id'] = self._id
        f.close()

    def cell_map(self):
        """
        Return a dataframe of cell ID's and locations
        """
        if 'cell_map' not in list(self.get_data('segmentation_images')['segmentation_label']): return None
        cmid = self.get_data('segmentation_images').set_index('segmentation_label').loc['cell_map','image_id']
        return map_image_ids(self.get_image(cmid)).rename(columns={'id':'cell_index'})

    def cell_map_image(self):
        """
        Return a the image of cells by ID's
        """
        if 'cell_map' not in list(self.get_data('segmentation_images')['segmentation_label']): return None
        cmid = self.get_data('segmentation_images').set_index('segmentation_label').loc['cell_map','image_id']
        return self.get_image(cmid)

    def edge_map(self):
        """
        Return a dataframe of cells by ID's of coordinates only on the edge of the cells
        """
        if 'edge_map' not in list(self.get_data('segmentation_images')['segmentation_label']): return None
        cmid = self.get_data('segmentation_images').set_index('segmentation_label').loc['edge_map','image_id']
        return map_image_ids(self.get_image(cmid)).\
                   rename(columns={'id':'cell_index'})

    def edge_map_image(self):
        """
        Return an image of edges of integers by ID
        """
        if 'edge_map' not in list(self.get_data('segmentation_images')['segmentation_label']): return None
        cmid = self.get_data('segmentation_images').set_index('segmentation_label').loc['edge_map','image_id']
        return self.get_image(cmid)

    def segmentation_info(self):
        """
        Return a dataframe with info about segmentation like cell areas and circumferences
        """
        
        # handle the case where there is no edge data
        if self.edge_map() is None:
            return pd.DataFrame(index=self.get_data('cells').index,columns=['edge_pixels','area_pixels'])

        return self.edge_map().reset_index().groupby(['cell_index']).count()[['x']].rename(columns={'x':'edge_pixels'}).\
            merge(self.cell_map().reset_index().groupby(['cell_index']).count()[['x']].rename(columns={'x':'area_pixels'}),
                  left_index=True,
                  right_index=True).reset_index().set_index('cell_index')
    def interaction_map(self):
        """
        Returns:
            pandas.DataFrame: return a dataframe of which cells are in contact with one another
        """
        return self.get_data('cell_interactions')
    def set_interaction_map(self,touch_distance=1):
        """
        Measure the cell-cell contact interactions

        Args:
            touch_distance (int): optional default is 1 distance to look away from a cell for another cell
        """
        full = self.cell_map()
        edge = self.edge_map()
        if full is None or edge is None: return None
        d1 = edge.reset_index()
        d1['key'] = 1
        d2 = pd.DataFrame({'mod':[-1*touch_distance,0,touch_distance]})
        d2['key'] = 1
        d3 = d1.merge(d2,on='key').merge(d2,on='key')
        d3['x'] = d3['x'].add(d3['mod_x'])
        d3['y'] = d3['y'].add(d3['mod_y'])
        d3 = d3[['x','y','cell_index','key']].rename(columns={'cell_index':'neighbor_cell_index'})
        im = full.reset_index().merge(d3,on=['x','y']).\
            query('cell_index!=neighbor_cell_index').\
            drop_duplicates().groupby(['cell_index','neighbor_cell_index']).count()[['key']].reset_index().\
            rename(columns={'key':'pixel_count'})
        im['touch_distance'] = touch_distance
        im.index.name='db_id'
        self.set_data('cell_interactions',im)

    #@property
    #def thresholds(self):
    #    raise ValueError('Override this to use it.')

    def get_channels(self,all=False):
        """
        Return a dataframe of the Channels

        Args:
            all (bool): default False if all is set to true will also include excluded channels (like autofluoresence)

        Returns:
            pandas.DataFrame: channel information
        """
        if all: return self.get_data('measurement_channels')
        d = self.get_data('measurement_channels')
        return d.loc[~d['channel_label'].isin(self.excluded_channels)]
    def get_regions(self):
        return self.get_data('regions')

    def get_labeled_raw(self,feature_label,statistic_label,all=False,channel_abbreviation=True):
        """
        Like get raw but add frame labels

        """
        df = self.get_raw(feature_label,statistic_label,all=all,channel_abbreviation=channel_abbreviation).reset_index()
        df['frame_name'] = self.frame_name
        df['frame_id'] = self.id
        return df.set_index(['frame_name','frame_id','cell_index'])



    def get_raw(self,measurement_feature_label,statistic_label,all=False,channel_abbreviation=True):
        """
        Get the raw data

        Args:
            feature_label (str): name of the feature
            statistic_label (str): name of the statistic to extract
            all (bool): default False if True put out everything including excluded channels
            channel_abbreviation (bool): default True means use the abbreivations if available

        Returns:
            pandas.DataFrame: the dataframe
        """
        stats = self.get_data('measurement_statistics').reset_index()
        stats = stats.loc[stats['statistic_label']==statistic_label,'statistic_index'].iloc[0]
        feat = self.get_data('measurement_features').reset_index()
        feat = feat.loc[feat['measurement_feature_label']==measurement_feature_label,'measurement_feature_index'].iloc[0]
        #region = self.get_data('regions').reset_index()
        #region = region.loc[region['region_label']==region_label,'region_index'].iloc[0]
        measure = self.get_data('cell_measurements')
        measure = measure.loc[(measure['statistic_index']==stats)&(measure['measurement_feature_index']==feat)]
        channels = self.get_data('measurement_channels')
        if not all: channels = channels.loc[~channels['channel_label'].isin(self.excluded_channels)]
        measure = measure.merge(channels,left_on='channel_index',right_index=True)
        measure = measure.reset_index().pivot(index='cell_index',columns='channel_label',values='value')
        if not channel_abbreviation: return measure
        temp = dict(zip(self.get_data('measurement_channels')['channel_label'],
                        self.get_data('measurement_channels')['channel_abbreviation']))
        return measure.rename(columns=temp)

    def default_raw(self):
        # override this
        return None

    def copy(self):
        mytype = type(self)
        them = mytype()
        for x in self.data_tables.keys():
            them._data[x] = self._data[x].copy()
        them._images = self._images.copy()
        them.set_processed_image_id(self._processed_image_id)
        them.frame_name = self.frame_name
        them._id = self._id
        return them

    @property
    def excluded_channels(self):
        raise ValueError("Must be overridden")

    def binary_calls(self):
        """
        Return all the binary feature calls (alias)
        """
        return phenotype_calls()

    def phenotype_calls(self):
        """
        Return all the binary feature calls
        """
        phenotypes = self.get_data('phenotypes')['phenotype_label'].dropna().tolist()
        temp = pd.DataFrame(index=self.get_data('cells').index,columns=phenotypes)
        temp = temp.fillna(0)
        temp = temp.merge(self.cell_df()[['phenotype_label']],left_index=True,right_index=True)
        for phenotype in phenotypes:
            temp.loc[temp['phenotype_label']==phenotype,phenotype]=1
        return temp.drop(columns='phenotype_label').astype(np.int8)

    def scored_calls(self):
        # Must be overridden
        return None
        

    def cdf(self,region_group=None,mutually_exclusive_phenotypes=None):
        """
        Return the pythologist.CellDataFrame of the frame

        Args:
            region_group (str): A region group present in the h5

        Returns:
            pythologist.CellDataFrame: the dataframe
        """
        
        _valid_region_group_names = sorted(self.get_data('region_groups')['region_group'].unique())
        if region_group is None or region_group not in _valid_region_group_names:
            raise ValueError("region_group must be one of the set regions: "+str(_valid_region_group_names))




        # get our region sizes
        _rgt = self.get_data('region_groups')
        _rgt = _rgt.loc[_rgt['region_group']==region_group,:].copy()
        if _rgt.shape[0] != 1:
            raise ValueError("Should only have one region group by this name")
        _rgt_index = _rgt.iloc[0,:].name
        _rt = self.get_data('regions')
        _rt = _rt.loc[_rt['region_group_index']==_rgt_index]
        region_count = _rt.groupby('region_label').count()['region_size']
        if region_count[region_count>1].shape[0]>0: 
            raise ValueError("duplicate region labels not supported") # add a saftey check
        region_sizes = _rt.set_index('region_label')['region_size'].astype(int).to_dict()


        # get our cells
        temp1 = self.get_data('cells').\
            merge(self.get_data('cell_regions'),
                  left_index=True,
                  right_on='cell_index').\
            merge(_rt,
                  left_on='region_index',
                  right_index=True).\
            drop(columns=['image_id','region_index','region_size','region_group_index'])
        temp1['regions'] = temp1.apply(lambda x: region_sizes,1)
        temp1['phenotype_label'] = 'TOTAL'
        temp1['phenotype_calls'] = temp1['phenotype_label'].apply(lambda x: {'TOTAL':1})


        _fdt = self.get_data('feature_definitions')
        _fdt['feature_tuple'] = _fdt.apply(lambda x: tuple([x['feature_value'],x['feature_value_label']]),1)
        _fgt = _fdt.groupby('feature_index').apply(lambda x: set(x['feature_tuple']))
        _binary_feature_index = _fgt.loc[_fgt=={(0, "-"), (1, "+")}].index
        # Filter to only binary features
        _fdt = _fdt.loc[_fdt['feature_index'].isin(_binary_feature_index),:].copy().drop(columns=['feature_tuple'])
        _fdt = self.get_data('features').merge(_fdt,left_index=True,right_on='feature_index').drop(columns=['feature_description'])
        _ft = _fdt.merge(self.get_data('cell_features'),left_index=True,right_on='feature_definition_index')
        _df = _ft.pivot(index='cell_index',columns='feature_label',values='feature_value').fillna(0).astype(int).\
            apply(lambda x: dict(zip(x.index,x)),1).reset_index().rename(columns={0:'scored_calls'})

        temp1 = temp1.merge(_df,on='cell_index')
        temp4 = self.default_raw()
        if temp4 is not None:
            temp4 = temp4.apply(lambda x:
                dict(zip(
                    list(x.index),
                    list(x)
                ))
            ,1).reset_index().rename(columns={0:'channel_values'}).set_index('cell_index')
            temp1 = temp1.merge(temp4,left_index=True,right_index=True)
        else:
            temp1['channel_values'] = np.nan
        #return temp1

        #temp5 = self.interaction_map().groupby('cell_index').\
        #    apply(lambda x: json.dumps(list(sorted(x['neighbor_cell_index'])))).reset_index().\
        #    rename(columns={0:'neighbor_cell_index'}).set_index('cell_index')


        # Get neighbor data .. may not be available for all cells
        #    Set a default of a null frame and only try and set if there are some neighbors present
        neighbors = pd.DataFrame(index=self.get_data('cells').index,columns=['neighbors'])
        if self.interaction_map().shape[0] > 0:
            neighbors = self.interaction_map().groupby('cell_index').\
                apply(lambda x:
                    dict(zip(
                        x['neighbor_cell_index'].astype(int),x['pixel_count'].astype(int)
                    ))
                ).reset_index().rename(columns={0:'neighbors'}).set_index('cell_index')

        # only do edges if we have them by setting a null value for default
        edge_length = pd.DataFrame(index=self.get_data('cells').index,columns=['edge_length'])
        if self.edge_map() is not None:
            edge_length = self.edge_map().reset_index().groupby('cell_index').count()[['x']].\
                rename(columns={'x':'edge_length'})
            edge_length['edge_length'] = edge_length['edge_length'].astype(int)

        cell_area = pd.DataFrame(index=self.get_data('cells').index,columns=['cell_area'])
        if self.cell_map() is not None:
            cell_area = self.cell_map().reset_index().groupby('cell_index').count()[['x']].\
                rename(columns={'x':'cell_area'})
            cell_area['cell_area'] = cell_area['cell_area'].astype(int)

        temp5 = cell_area.merge(edge_length,left_index=True,right_index=True).merge(neighbors,left_index=True,right_index=True,how='left')
        temp5.loc[temp5['neighbors'].isna(),'neighbors'] = temp5.loc[temp5['neighbors'].isna(),'neighbors'].apply(lambda x: {}) # these are ones we actuall have measured

        # If we DO have cell_map, merge in
        if self.cell_map() is not None:
            temp1 = temp1.drop(columns=['cell_area','edge_length']).merge(temp5,left_index=True,right_index=True,how='left')
        else:
            temp1 = temp1.merge(temp5.drop(columns=['cell_area','edge_length']),left_index=True,right_index=True,how='left')
        temp1.loc[temp1['neighbors'].isna(),'neighbors'] = np.nan # These we were not able to measure


        temp1['frame_name'] = self.frame_name
        temp1['frame_id'] = self.id
        #temp1  = temp1.reset_index()
        temp1 = temp1.sort_values('cell_index').reset_index(drop=True)
        temp1['sample_name'] = 'undefined'
        temp1['project_name'] = 'undefined'
        temp1['sample_id'] = 'undefined'
        temp1['project_id'] = 'undefined'
        def _get_phenotype(d):
            if d!=d: return np.nan # set to null if there is nothing in phenotype calls
            vals = [k for k,v in d.items() if v ==  1]
            return np.nan if len(vals) == 0 else vals[0]
        temp1['phenotype_label'] = temp1.apply(lambda x:
                  _get_phenotype(x['phenotype_calls'])
            ,1)
        # Let's tack on the image shape
        temp1['frame_shape'] = temp1.apply(lambda x: self.shape,1)
        cdf = CellDataFrame(temp1)
        if mutually_exclusive_phenotypes is not None:
            cdf = cdf.scored_to_phenotypes(mutually_exclusive_phenotypes).drop_scored_calls(mutually_exclusive_phenotypes)
        return cdf

    def binary_df(self):
        temp1 = self.phenotype_calls().stack().reset_index().\
            rename(columns={'level_1':'binary_phenotype',0:'score'})
        temp1.loc[temp1['score']==1,'score'] = '+'
        temp1.loc[temp1['score']==0,'score'] = '-'
        temp1['gated'] = 0
        temp1.index.name = 'db_id'
        return temp1

    def cell_df(self):
        celldf = self.get_data('cells').\
            merge(self.get_data('regions').rename(columns={'image_id':'region_image_id'}),
                  left_on='region_index',
                  right_index=True).\
            merge(self.get_data('phenotypes'),left_on='phenotype_index',right_index=True).\
            merge(self.segmentation_info(),left_index=True,right_index=True,how='left')
        return celldf.drop(columns=['phenotype_index','region_index'])

    def complete_df(self):
        # a dataframe for every cell that has everything
        return

    def get_image(self,image_id):
        """
        Args:
            image_id (str): get the image by this id

        Returns:
            numpy.array: an image representing a 2d array
        """
        return self._images[image_id].copy()

class CellSampleGeneric(object):
    def __init__(self):
        self._frames = {}
        self._key = None
        self._id = uuid4().hex
        self.sample_name = np.nan
        return

    @property
    def id(self):
        """
        Return the UUID4 str
        """
        return self._id

    def create_cell_frame_class(self):
        return CellFrameGeneric()

    @property
    def frame_ids(self):
        """
        Return the list of frame IDs
        """
        return sorted(list(self._frames.keys()))

    @property
    def key(self):
        """
        Return a pandas.DataFrame of info about the sample
        """
        return self._key

    def get_frame(self,frame_id):
        """
        Args:
            frame_id (str): the ID of the frame you want to access

        Returns:
            CellFrameGeneric: the cell frame
        """
        return self._frames[frame_id]

    def cdf(self,region_group=None,mutually_exclusive_phenotypes=None):
        """
        Return the pythologist.CellDataFrame of the sample
        """
        output = []
        for frame_id in self.frame_ids:
            temp = self.get_frame(frame_id).cdf(region_group=region_group,mutually_exclusive_phenotypes=mutually_exclusive_phenotypes)
            temp['sample_name'] = self.sample_name
            temp['sample_id'] = self.id
            output.append(temp)
        output = pd.concat(output).reset_index(drop=True)
        output.index.name = 'db_id'
        output['project_name'] = 'undefined'
        output['project_id'] = 'undefined'
        return CellDataFrame(pd.DataFrame(output))


    def to_hdf(self,h5file,location='',mode='w'):
        #print(mode)
        f = h5py.File(h5file,mode)
        #f.create_group(location+'/meta')
        #f.create_dataset(location+'/meta/id',data=self.id)
        #f.create_dataset(location+'/meta/sample_name',data=self.sample_name)
        if location+'/meta' in f:
            del f[location+'/meta']
        dset = f.create_dataset(location+'/meta', (100,), dtype=h5py.special_dtype(vlen=str))
        dset.attrs['sample_name'] = self.sample_name
        dset.attrs['id'] = self._id
        if location+'/frames' in f:
            del f[location+'/frames']
        f.create_group(location+'/frames')
        f.close()
        for frame_id in self.frame_ids:
            frame = self._frames[frame_id]
            frame.to_hdf(h5file,
                         location+'/frames/'+frame_id,
                          mode='a')
        self._key.to_hdf(h5file,location+'/info',mode='r+',format='table',complib='zlib',complevel=9)


    def read_hdf(self,h5file,location=''):
        if location != '': location = location.split('/')
        else: location = []
        f = h5py.File(h5file,'r')
        subgroup = f
        for x in location:
            subgroup = subgroup[x]
        self._id = subgroup['meta'].attrs['id']
        self.sample_name = subgroup['meta'].attrs['sample_name']
        frame_ids = [x for x in subgroup['frames']]
        for frame_id in frame_ids:
            cellframe = self.create_cell_frame_class()
            loc = '/'.join(location+['frames',frame_id])
            #print(loc)
            cellframe.read_hdf(h5file,location=loc)
            self._frames[frame_id] = cellframe
            #self.frame_name = str(subgroup['frames'][frame_id]['meta']['frame_name'])
            #self._id = str(subgroup['frames'][frame_id]['meta']['id'])
        loc = '/'.join(location+['info'])
        #print(loc)
        self._key = pd.read_hdf(h5file,loc)
        f.close()
        return

    def cell_df(self):
        frames = []
        for frame_id in self.frame_ids:
            frame = self.get_frame(frame_id).cell_df().reset_index()
            key_line = self.key.set_index('frame_id').loc[[frame_id]].reset_index()
            key_line['key'] = 1
            frame['key'] = 1
            frame = key_line.merge(frame,on='key').drop(columns = 'key')
            frames.append(frame)
        frames = pd.concat(frames).reset_index(drop=True)
        frames.index.name = 'sample_cell_index'
        return frames

    def binary_df(self):
        fc = self.cell_df()[['frame_id','cell_index']].reset_index()
        frames = []
        for frame_id in self.frame_ids:
            frame = self.get_frame(frame_id).binary_df()
            key_line = self.key.set_index('frame_id').loc[[frame_id]].reset_index()
            key_line['key'] = 1
            frame['key'] = 1
            frame = key_line.merge(frame,on='key').drop(columns = 'key')
            frames.append(frame)
        return fc.merge(pd.concat(frames).reset_index(drop=True),on=['frame_id','cell_index'])

    def interaction_map(self):
        fc = self.cell_df()[['frame_id','cell_index']].reset_index()
        frames = []
        for frame_id in self.frame_ids:
            frame = self.get_frame(frame_id).interaction_map()
            key_line = self.key.set_index('frame_id').loc[[frame_id]].reset_index()
            key_line['key'] = 1
            frame['key'] = 1
            frame = key_line.merge(frame,on='key').drop(columns = 'key')
            frames.append(frame)
        frames = pd.concat(frames).reset_index(drop=True)
        return frames.merge(fc,on=['frame_id','cell_index']).\
                      merge(fc.rename(columns={'sample_cell_index':'neighbor_sample_cell_index',
                                               'cell_index':'neighbor_cell_index'}),
                            on=['frame_id','neighbor_cell_index'])
    def frame_iter(self):
        """
        An iterator of frames

        Returns:
            CellFrameGeneric
        """
        for frame_id in self.frame_ids:
            yield self.get_frame(frame_id)
   
    def get_labeled_raw(self,feature_label,statistic_label,all=False,channel_abbreviation=True):
        """
        Return a matrix of raw data labeled by samples and frames names/ids

        Returns:
            DataFrame
        """        
        vals = []
        for f in self.frame_iter():
            df = f.get_labeled_raw(feature_label,statistic_label,all=all,channel_abbreviation=channel_abbreviation).reset_index()
            df['sample_name'] = self.sample_name
            df['sample_id'] = self.id
            vals.append(df)
        return pd.concat(vals).set_index(['sample_name','sample_id','frame_name','frame_id','cell_index'])

class CellProjectGeneric(object):
    def __init__(self,h5path,mode='r'):
        """
        Create a CellProjectGeneric object or read from/add to an existing one

        Args:
            h5path (str): path to read/from or store/to, only use None for a dry_run read
            mode (str): 'r' read, 'a' append, 'w' create/write, 'r+' create/append if necessary
        """
        self._key = None
        self.h5path = h5path if h5path is not None else SpooledTemporaryFile(max_size=100*1e6)
        self.mode = mode
        self._sample_cache_name = None
        self._sample_cache = None
        if mode =='r':
            if not os.path.exists(h5path): raise ValueError("Cannot read a file that does not exist")
        if mode == 'w' or mode == 'r+':
            f = h5py.File(self.h5path,mode)
            if '/samples' not in f.keys():
                f.create_group('/samples')
            if '/meta' not in f.keys():
                dset = f.create_dataset('/meta', (100,), dtype=h5py.special_dtype(vlen=str))
            else:
                dset = f['/meta']
            dset.attrs['project_name'] = np.nan
            dset.attrs['microns_per_pixel'] = np.nan
            dset.attrs['id'] = uuid4().hex
            f.close()
        return

    def copy(self,path,overwrite=False,output_mode='r'):
        if os.path.exists(path) and overwrite is False: 
            raise ValueError("Cannot overwrite unless overwrite is set to True")
        shutil.copy(self.h5path,path)
        return self.__class__(path,mode=output_mode)

    def to_hdf(self,path,overwrite=False):
        """
        Write this object to another h5 file
        """
        self.copy(path,overwrite=overwrite)
        return


    @classmethod
    def concat(self,path,array_like,overwrite=False,verbose=False):
        if os.path.exists(path) and overwrite is False: 
            raise ValueError("Cannot overwrite unless overwrite is set to True")
        # copy the first 
        arr = [x for x in array_like]
        if len(arr) == 0: raise ValueError("cannot concat empty list")
        if verbose: sys.stderr.write("Copy the first element\n")
        cpi = arr[0].copy(path,output_mode='r+',overwrite=overwrite)
        #shutil.copy(arr[0].h5path,path)
        #cpi = CellProjectGeneric(path,mode='r+')
        if len(arr) == 1: return 
        for project in array_like[1:]:
            if verbose: sys.stderr.write("Add project "+str(project.id)+" "+str(project.project_name)+"\n")
            for s in project.sample_iter():
                if verbose: sys.stderr.write("   Add sample "+str(s.id)+" "+str(s.sample_name)+"\n")
                cpi.append_sample(s)
        return cpi

    def append_sample(self,sample):
        """
        Append sample to the project

        Args:
            sample (CellSampleGeneric): sample object
        """
        if self.mode == 'r': raise ValueError("Error: cannot write to a path in read-only mode.")
        sample.to_hdf(self.h5path,location='samples/'+sample.id,mode='a')
        
        current = self.key
        if current is None:
            current = pd.DataFrame([{'sample_id':sample.id,
                                     'sample_name':sample.sample_name}])
            current.index.name = 'db_id'
        else:
            iteration = max(current.index)+1
            addition = pd.DataFrame([{'db_id':iteration,
                                      'sample_id':sample.id,
                                      'sample_name':sample.sample_name}]).set_index('db_id')
            current = pd.concat([current,addition])
        current.to_hdf(self.h5path,'info',mode='r+',complib='zlib',complevel=9,format='table')
        return

    def qc(self,*args,**kwargs):
        """
        Returns:
            QC: QC class to do quality checks
        """
        return QC(self,*args,**kwargs)

    @property
    def id(self):
        """
        Returns the (str) UUID4 string
        """
        f = h5py.File(self.h5path,'r')
        name = f['meta'].attrs['id']
        f.close()
        return name

    @property 
    def project_name(self):
        """
        Return or set the (str) project_name
        """
        f = h5py.File(self.h5path,'r')
        name = f['meta'].attrs['project_name']
        f.close()
        return name
    @project_name.setter
    def project_name(self,name):
        if self.mode == 'r': raise ValueError('cannot write if read only')
        f = h5py.File(self.h5path,'r+')
        f['meta'].attrs['project_name'] = name
        f.close()

    @property 
    def microns_per_pixel(self):
        """
        Return or set the (float) microns_per_pixel
        """
        f = h5py.File(self.h5path,'r')
        name = f['meta'].attrs['microns_per_pixel']
        f.close()
        return name
    @microns_per_pixel.setter
    def microns_per_pixel(self,value):
        if self.mode == 'r': raise ValueError('cannot write if read only')
        f = h5py.File(self.h5path,'r+')
        f['meta'].attrs['microns_per_pixel'] = value
        f.close()

    def set_id(self,name):
        """
        Set the project ID

        Args:
            name (str): project_id
        """
        if self.mode == 'r': raise ValueError('cannot write if read only')
        f = h5py.File(self.h5path,'r+')
        #dset = f.create_dataset('/meta', (100,), dtype=h5py.special_dtype(vlen=str))
        f['meta'].attrs['id'] = name
        f.close()

    def cdf(self,region_group=None,mutually_exclusive_phenotypes=None):
        """
        Return the pythologist.CellDataFrame of the project
        """
        output = []
        for sample_id in self.sample_ids:
            temp = self.get_sample(sample_id).cdf(region_group=region_group,mutually_exclusive_phenotypes=mutually_exclusive_phenotypes)
            temp['project_name'] = self.project_name
            temp['project_id'] = self.id
            output.append(temp)
        output = pd.concat(output).reset_index(drop=True)
        output.index.name = 'db_id'
        cdf = CellDataFrame(pd.DataFrame(output))
        if self.microns_per_pixel: cdf.microns_per_pixel = self.microns_per_pixel
        return cdf

    def cell_df(self):
        samples = []
        for sample_id in self.sample_ids:
            sample = self.get_sample(sample_id).cell_df().reset_index()
            key_line = self.key.set_index('sample_id').loc[[sample_id]].reset_index()
            key_line['key'] = 1
            sample['key'] = 1
            sample = key_line.merge(sample,on='key').drop(columns = 'key')
            samples.append(sample)
        samples = pd.concat(samples).reset_index(drop=True)
        samples.index.name = 'project_cell_index'
        return samples

    def binary_df(self):
        fc = self.cell_df()[['sample_id','frame_id','cell_index']].reset_index()
        samples = []
        for sample_id in self.sample_ids:
            sample = self.get_sample(sample_id).binary_df()
            key_line = self.key.set_index('sample_id').loc[[sample_id]].reset_index()
            key_line['key'] = 1
            sample['key'] = 1
            sample = key_line.merge(sample,on='key').drop(columns = 'key')
            samples.append(sample)
        return fc.merge(pd.concat(samples).reset_index(drop=True),on=['sample_id','frame_id','cell_index'])
    
    def interaction_map(self):
        fc = self.cell_df()[['sample_id','frame_id','cell_index']].reset_index()
        samples = []
        for sample_id in self.sample_ids:
            sample = self.get_sample(sample_id).interaction_map()
            key_line = self.key.set_index('sample_id').loc[[sample_id]].reset_index()
            key_line['key'] = 1
            sample['key'] = 1
            sample = key_line.merge(sample,on='key').drop(columns = 'key')
            samples.append(sample)
        samples = pd.concat(samples).reset_index(drop=True)
        return samples.merge(fc,on=['sample_id','frame_id','cell_index']).\
                       merge(fc.rename(columns={'project_cell_index':'neighbor_project_cell_index',
                                               'cell_index':'neighbor_cell_index'}),
                             on=['sample_id','frame_id','neighbor_cell_index'])


    def create_cell_sample_class(self):
        return CellSampleGeneric()
    
    @property
    def sample_ids(self):
        """
        Return the list of sample_ids
        """
        return sorted(list(self.key['sample_id']))

    def get_sample(self,sample_id):
        """
        Get the sample_id

        Args:
            sample_id (str): set the sample id
        """
        if self._sample_cache_name == sample_id:
            return self._sample_cache
        sample = self.create_cell_sample_class()
        sample.read_hdf(self.h5path,'samples/'+sample_id)
        self._sample_cache_name = sample_id
        self._sample_cache = sample
        return sample

    @property
    def key(self):
        """
        Get info about the project
        """
        f = h5py.File(self.h5path,'r')
        val = False
        if 'info' in [x for x in f]: val = True
        f.close()
        return None if not val else pd.read_hdf(self.h5path,'info')
    
    def sample_iter(self):
        """
        An interator of CellSampleGeneric
        """
        for sample_id in self.sample_ids: yield self.get_sample(sample_id)

    def frame_iter(self):
        """
        An interator of CellFrameGeneric
        """
        for s in self.sample_iter():
            for frame_id in s.frame_ids:
                yield s.get_frame(frame_id)

    @property
    def channel_image_dataframe(self):
        """
        dataframe within info about channels and images
        """
        pname = self.project_name
        pid = self.id
        measurements = []
        for s in self.sample_iter():
            sname = s.sample_name
            sid = s.id
            for f in s.frame_iter():
                fname = f.frame_name
                fid = f.id
                mc = f.get_data('measurement_channels')
                mc['project_name'] = pname
                mc['project_id'] = pid
                mc['sample_name'] = sname
                mc['sample_id'] = sid
                mc['frame_name'] = fname
                mc['frame_id'] = fid
                mc['processed_image_id'] = f.processed_image_id
                measurements.append(mc)
        return pd.concat(measurements).reset_index(drop=True)

    def get_image(self,sample_id,frame_id,image_id):
        """
        Get an image by sample frame and image id

        Args:
            sample_id (str): unique sample id
            frame_id (str): unique frame id
            image_id (str): unique image id

        Returns:
            numpy.array: 2d image array
        """
        s = self.get_sample(sample_id)
        f = s.get_frame(frame_id)
        return f.get_image(image_id)

    def get_labeled_raw(self,feature_label,statistic_label,all=False,channel_abbreviation=True):
        """
        Return a matrix of raw data labeled by samples and frames names/ids

        Returns:
            DataFrame
        """        
        vals = []
        for s in self.sample_iter():
            df = s.get_labeled_raw(feature_label,statistic_label,all=all,channel_abbreviation=channel_abbreviation).reset_index()
            df['project_name'] = self.project_name
            df['project_id'] = self.id
            vals.append(df)
        return pd.concat(vals).set_index(['project_name','project_id','sample_name','sample_id','frame_name','frame_id','cell_index'])    
