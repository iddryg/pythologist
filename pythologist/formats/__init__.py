import pandas as pd
import numpy as np
import h5py, os
from uuid import uuid4
from pythologist.formats.utilities import map_image_ids
""" These are classes to help deal with cell-level image data """

class CellImageGeneric(object):
    """ A generic CellImageData object
    """
    def __init__(self):
        # Define the column structure of all the tables.  
        #   Non-generic CellImageData may define additional data tables
        self._processed_image_id = None
        self._images = {}                      # Database of Images
        self._id = uuid4().hex
        self.data_tables = {
        'cells':{'index':'cell_index',            
                  'columns':['x','y','phenotype_index',
                             'region_index']},
        'cell_tags':{'index':'db_id',            
                     'columns':['tag_index','cell_index']},
        'cell_measurements':{'index':'measurement_index', 
                             'columns':['cell_index','statistic_index','feature_index','channel_index','value']},
        'measurement_features':{'index':'feature_index',
                                'columns':['feature_label']},
        'measurement_channels':{'index':'channel_index',
                                'columns':['channel_label','channel_abbreviation','image_id']},
        'measurement_statistics':{'index':'statistic_index',
                                  'columns':['statistic_label']},
        'phenotypes':{'index':'phenotype_index',
                      'columns':['phenotype_label']},
        'segmentation_images':{'index':'db_id',
                 'columns':['segmentation_label','image_id']},                     
        'regions':{'index':'region_index',
                   'columns':['region_label','region_size','image_id']},
        'cell_interactions':{'index':'db_id', 
                             'columns':['cell_index','neighbor_cell_index','pixel_count','touch_distance']},
        'tags':{'index':'tag_index',
                'columns':['tag_label']}
                           }
        self._data = {} # Do not acces directly. Use set_data_table and get_data_table to access.
        for x in self.data_tables.keys(): 
            self._data[x] = pd.DataFrame(columns=self.data_tables[x]['columns'])
            self._data[x].index.name = self.data_tables[x]['index']
    @property
    def id(self):
        return self._id
    def set_data(self,table_name,table):
        # Assign data to the standard tables. Do some column name checking to make sure we are getting what we expect
        if table_name not in self.data_tables: raise ValueError("Error table name doesn't exist in defined formats")
        if set(list(table.columns)) != set(self.data_tables[table_name]['columns']): raise ValueError("Error column names don't match defined format\n"+\
                                                                                            str(list(table.columns))+"\n"+\
                                                                                            str(self.data_tables[table_name]['columns']))
        if table.index.name != self.data_tables[table_name]['index']: raise ValueError("Error index name doesn't match defined format")
        self._data[table_name] = table.loc[:,self.data_tables[table_name]['columns']].copy() # Auto-sort, and assign a copy so we aren't ever assigning by reference

    def get_data(self,table_name): 
        # copy so we don't ever pass by reference
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
        return

    def to_hdf(self,h5file,location='',mode='w'):
        f = h5py.File(h5file,mode)
        f.create_group(location+'/data')
        f.create_group(location+'/images')
        f.close()
        for table_name in self.data_tables.keys():
            data_table = self.get_data(table_name)
            data_table.to_hdf(h5file,
                              location+'/data/'+table_name,
                              mode='r+',
                              format='table',
                              complib='zlib',
                              complevel=9)
        f = h5py.File(h5file,'a')
        for image_id in self._images.keys():
            f.create_dataset(location+'/images/'+image_id,data=self._images[image_id],compression='gzip')
        f.close()

    def cell_map(self):
        if 'cell_map' not in list(self.get_data('segmentation_images')['segmentation_label']): return None
        cmid = self.get_data('segmentation_images').set_index('segmentation_label').loc['cell_map','image_id']
        return map_image_ids(self.get_image(cmid)).rename(columns={'id':'cell_index'}).set_index('cell_index')

    def edge_map(self):
        if 'edge_map' not in list(self.get_data('segmentation_images')['segmentation_label']): return None
        cmid = self.get_data('segmentation_images').set_index('segmentation_label').loc['edge_map','image_id']
        return map_image_ids(self.get_image(cmid)).\
                   rename(columns={'id':'cell_index'}).set_index('cell_index')

    def segmentation_info(self):
        return self.edge_map().reset_index().groupby(['cell_index']).count()[['x']].rename(columns={'x':'edge_pixels'}).\
            merge(self.cell_map().reset_index().groupby(['cell_index']).count()[['x']].rename(columns={'x':'area_pixels'}),
                  left_index=True,
                  right_index=True).reset_index().set_index('cell_index')
    def interaction_map(self):
        return self.get_data('cell_interactions')
    def set_interaction_map(self,touch_distance=1):
        full = self.cell_map()
        edge = self.edge_map()
        if full is None or edge is None: return None
        d1 = edge.reset_index()
        d1['key'] = 1
        d2 = pd.DataFrame({'mod':[-1*touch_distance,0,touch_distance]})
        d2['key'] = 1
        d3 = d1.merge(d2,on='key').merge(d2,on='key')
        d3['x'] = d3.apply(lambda x: x['x']+x['mod_x'],1)
        d3['y'] = d3.apply(lambda x: x['y']+x['mod_y'],1)
        d3 = d3[['x','y','cell_index','key']].rename(columns={'cell_index':'neighbor_cell_index'})
        im = full.reset_index().merge(d3,on=['x','y']).\
            query('cell_index!=neighbor_cell_index').\
            drop_duplicates().groupby(['cell_index','neighbor_cell_index']).count()[['key']].reset_index().\
            rename(columns={'key':'pixel_count'})
        im['touch_distance'] = touch_distance
        im.index.name='db_id'
        self.set_data('cell_interactions',im)

    @property
    def thresholds(self):
        raise ValueError('Override this to use it.')

    def get_channels(self,all=False):
        if all: return self.get_data('measurement_channels')
        d = self.get_data('measurement_channels')
        return d.loc[~d['channel_label'].isin(self.excluded_channels)]
    def get_regions(self):
        return self.get_data('regions')
    
    def get_raw(self,feature_label,statistic_label,region_label,all=False,channel_abbreviation=True):
        stats = self.get_data('measurement_statistics').reset_index()
        stats = stats.loc[stats['statistic_label']==statistic_label,'statistic_index'].iloc[0]
        feat = self.get_data('measurement_features').reset_index()
        feat = feat.loc[feat['feature_label']==feature_label,'feature_index'].iloc[0]
        region = self.get_data('regions').reset_index()
        region = region.loc[region['region_label']==region_label,'region_index'].iloc[0]
        measure = self.get_data('cell_measurements')
        measure = measure.loc[(measure['statistic_index']==stats)&(measure['feature_index']==feat)]
        channels = self.get_data('measurement_channels')
        if not all: channels = channels.loc[~channels['channel_label'].isin(self.excluded_channels)]
        measure = measure.merge(channels,left_on='channel_index',right_index=True)
        measure = measure.reset_index().pivot(index='cell_index',columns='channel_label',values='value')
        if not channel_abbreviation: return measure
        temp = dict(zip(self.get_data('measurement_channels')['channel_label'],
                        self.get_data('measurement_channels')['channel_abbreviation']))
        return measure.rename(columns=temp)

    def copy(self):
        # Do a deep copy of self
        mytype = type(self)
        them = mytype()
        for x in self.data_tables.keys():
            them._data[x] = self._data[x].copy()
        return them

    @property
    def excluded_channels(self):
        raise ValueError("Must be overidden")

    def binary_calls(self):
        # Default to just gating on mutually exclusive phenotypes
        phenotypes = self.get_data('phenotypes')['phenotype_label'].dropna().tolist()
        temp = pd.DataFrame(index=self.get_data('cells').index,columns=phenotypes)
        temp = temp.fillna(0)
        temp = temp.merge(self.cell_df()[['phenotype_label']],left_index=True,right_index=True)
        for phenotype in phenotypes:
            temp.loc[temp['phenotype_label']==phenotype,phenotype]=1
        return temp.drop(columns='phenotype_label').astype(np.int8)

    #@property
    #def df(self):
    #    # a dataframe that has phenotype and region info, but excludes all raw data
    #    return self.get_data('cells').merge(self.get_data('phenotypes'),left_on='phenotype_index',right_index=True,how='left').drop(columns='phenotype_index').\
    #                                  merge(self.get_data('regions'),left_on='region_index',right_index=True,how='left').drop(columns='region_index').sort_index()

    def binary_df(self):
        binarydf = self.binary_calls().stack().reset_index().\
            rename(columns={'level_1':'phenotype',0:'score'})
        binarydf.loc[binarydf['score']==0,'score'] = '-'
        binarydf.loc[binarydf['score']==1,'score'] = '+'
        binarydf['score'] = binarydf['score'].astype(str)
        return binarydf

    def cell_df(self):
        celldf = self.get_data('cells').\
            merge(self.get_data('regions').rename(columns={'image_id':'region_image_id'}),
                  left_on='region_index',
                  right_index=True).\
            merge(self.get_data('phenotypes'),left_on='phenotype_index',right_index=True).\
            merge(self.segmentation_info(),left_index=True,right_index=True,how='left')
        return celldf

    def complete_df(self):
        # a dataframe for every cell that has everything
        return

    def get_image(self,image_id):
        return self._images[image_id].copy()

class CellSampleGeneric(object):
    def __init__(self):
        self._frames = {}
        self._key = None
        return

    @staticmethod
    def create_cell_image_class():
        return CellImageGeneric()

    @property
    def sample_names(self):
        return list(self._frames.keys())

    @property
    def key(self):
        return self._key

    def get_frame(self,frame_id):
        return self._frames[frame_id]

    def to_hdf(self,h5file,location='',mode='w'):
        f = h5py.File(h5file,mode)
        f.create_group(location+'/frames')
        #f.create_group(location+'/meta')
        f.close()
        for frame_id in self._frames.keys():
            frame = self._frames[frame_id]
            frame.to_hdf(h5file,
                         location+'/frames/'+frame_id,
                          mode='a')
        self._key.to_hdf(h5file,location+'/meta',mode='r+',format='table',complib='zlib',complevel=9)

    def read_hdf(self,h5file,location=''):
        if location != '': location = location.split('/')
        else: location = []
        f = h5py.File(h5file,'r')
        subgroup = f
        for x in location:
            subgroup = subgroup[x]
        frame_ids = [x for x in subgroup['frames']]
        for frame_id in frame_ids:
            cellimage = self.create_cell_image_class()
            loc = '/'.join(location+['frames',frame_id])
            #print(loc)
            self._frames[frame_id] = cellimage.read_hdf(h5file,location=loc)
        loc = '/'.join(location+['meta'])
        #print(loc)
        self._key = pd.read_hdf(h5file,loc)
        return

    def cell_df(self):
        frames = []
        for frame_id in sorted(list(self.key['frame_id'])):
            frame = self.get_frame(frame_id).cell_df().reset_index()
            key_line = self.key.set_index('frame_id').loc[[frame_id]].reset_index()
            key_line['key'] = 1
            frame['key'] = 1
            frame = key_line.merge(frame,on='key').drop(columns = 'key')
            frames.append(frame)
        frames = pd.concat(frames).reset_index(drop=True)
        frames.index.name = 'frame_cell_index'
        return frames

    def binary_df(self):
        fc = self.cell_df()[['frame_id','cell_index']].reset_index()
        frames = []
        for frame_id in self.key['frame_id']:
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
        for frame_id in self.key['frame_id']:
            frame = self.get_frame(frame_id).interaction_map()
            key_line = self.key.set_index('frame_id').loc[[frame_id]].reset_index()
            key_line['key'] = 1
            frame['key'] = 1
            frame = key_line.merge(frame,on='key').drop(columns = 'key')
            frames.append(frame)
        frames = pd.concat(frames).reset_index(drop=True)
        return frames.merge(fc,on=['frame_id','cell_index']).\
                      merge(fc.rename(columns={'frame_cell_index':'neighbor_frame_cell_index',
                                               'cell_index':'neighbor_cell_index'}),
                            on=['frame_id','neighbor_cell_index'])
    