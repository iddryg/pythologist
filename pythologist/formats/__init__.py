import pandas as pd
import numpy as np
import h5py, os, json, sys
from uuid import uuid4
from pythologist.formats.utilities import map_image_ids
from pythologist import CellDataFrame

""" These are classes to help deal with cell-level image data """

class CellFrameGeneric(object):
    """ A generic CellFrameData object
    """
    def __init__(self):
        # Define the column structure of all the tables.  
        #   Non-generic CellFrameData may define additional data tables
        self._processed_image_id = None
        self._images = {}                      # Database of Images
        self._id = uuid4().hex
        self.frame_name = None
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

    @property
    def processed_image_id(self):
        return self._processed_image_id
    @property
    def processed_image(self):
        return self._images[self._processed_image_id].copy()
    def set_processed_image_id(self,image_id):
        self._processed_image_id = image_id

    @property
    def table_names(self):
        return list(self.data_tables.keys())

    def set_data(self,table_name,table):
        # Assign data to the standard tables. Do some column name checking to make sure we are getting what we expect
        if table_name not in self.data_tables: raise ValueError("Error table name doesn't exist in defined formats")
        if set(list(table.columns)) != set(self.data_tables[table_name]['columns']): raise ValueError("Error column names don't match defined format\n"+\
                                                                                            str(list(table.columns))+"\n"+\
                                                                                            str(self.data_tables[table_name]['columns']))
        if table.index.name != self.data_tables[table_name]['index']: raise ValueError("Error index name doesn't match defined format")
        self._data[table_name] = table.loc[:,self.data_tables[table_name]['columns']].copy() # Auto-sort, and assign a copy so we aren't ever assigning by reference

    def set_regions(self,regions,use_processed_region=True,unset_label='undefined'):
        """
        Alter the regions in the frame

        regions: a dictionary of mutually exclusive region labels and binary masks
                 if a region does not cover all the workable areas then it will be the only label
                 and the unused area will get the 'unset_label' as a different region
        """
        
        # delete our current regions

        regions = regions.copy()
        image_ids = list(self.get_data('mask_images')['image_id'])
        image_ids = [x for x in image_ids if x != self.processed_image_id]
        for image_id in image_ids: del self._images[image_id]

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

        sys.stderr.write("Remaining areas after setting are "+str(remainder.sum())+"\n")

        if remainder.sum() > 0:
            labels += [unset_label]
            sizes += [remainder.sum()]
            ids += [uuid4().hex]
            self._images[ids[-1]] = remainder
            regions[unset_label] = remainder

        regions2 = pd.DataFrame({'region_label':labels,
                                 'region_size':sizes,
                                 'image_id':ids
                                })
        regions2.index.name = 'region_index'
        self.set_data('regions',regions2)

        def get_label(x,y):
            for label in regions:
                if regions[label][y][x] == 1: return label
            raise ValueError("Coordinate is out of bounds for all regions.")
        recode = self.get_data('cells').copy()
        recode['new_region_label'] = recode.apply(lambda x: get_label(x['x'],x['y']),1)
        recode = recode.drop(columns='region_index').reset_index().\
            merge(regions2[['region_label']].reset_index(),
                  left_on='new_region_label',right_on='region_label').\
            drop(columns=['region_label','new_region_label']).set_index('cell_index')
        self.set_data('cells',recode)
        return


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
            f.create_dataset(location+'/images/'+image_id,data=self._images[image_id],compression='gzip')
        dset = f.create_dataset(location+'/meta', (100,), dtype=h5py.special_dtype(vlen=str))
        dset.attrs['frame_name'] = self.frame_name
        dset.attrs['processed_image_id'] = self.processed_image_id
        dset.attrs['id'] = self._id
        #f.create_dataset(location+'/meta/frame_name',(len(self.frame_name),),dtype='S10')
        #f[location+'meta/frame_name'].attrs['frame_name'] = self.frame_name
        #f.create_dataset(location+'/meta/id',data=self.id)
        
        f.close()

    def cell_map(self):
        if 'cell_map' not in list(self.get_data('segmentation_images')['segmentation_label']): return None
        cmid = self.get_data('segmentation_images').set_index('segmentation_label').loc['cell_map','image_id']
        return map_image_ids(self.get_image(cmid)).rename(columns={'id':'cell_index'})

    def cell_map_image(self):
        if 'cell_map' not in list(self.get_data('segmentation_images')['segmentation_label']): return None
        cmid = self.get_data('segmentation_images').set_index('segmentation_label').loc['cell_map','image_id']
        return self.get_image(cmid)

    def edge_map(self):
        if 'edge_map' not in list(self.get_data('segmentation_images')['segmentation_label']): return None
        cmid = self.get_data('segmentation_images').set_index('segmentation_label').loc['edge_map','image_id']
        return map_image_ids(self.get_image(cmid)).\
                   rename(columns={'id':'cell_index'})

    def edge_map_image(self):
        if 'edge_map' not in list(self.get_data('segmentation_images')['segmentation_label']): return None
        cmid = self.get_data('segmentation_images').set_index('segmentation_label').loc['edge_map','image_id']
        return self.get_image(cmid)

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

    @property
    def thresholds(self):
        raise ValueError('Override this to use it.')

    def get_channels(self,all=False):
        if all: return self.get_data('measurement_channels')
        d = self.get_data('measurement_channels')
        return d.loc[~d['channel_label'].isin(self.excluded_channels)]
    def get_regions(self):
        return self.get_data('regions')
    
    def get_raw(self,feature_label,statistic_label,all=False,channel_abbreviation=True):
        stats = self.get_data('measurement_statistics').reset_index()
        stats = stats.loc[stats['statistic_label']==statistic_label,'statistic_index'].iloc[0]
        feat = self.get_data('measurement_features').reset_index()
        feat = feat.loc[feat['feature_label']==feature_label,'feature_index'].iloc[0]
        #region = self.get_data('regions').reset_index()
        #region = region.loc[region['region_label']==region_label,'region_index'].iloc[0]
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

    def default_raw(self):
        # override this
        return None

    def copy(self):
        # Do a deep copy of self
        mytype = type(self)
        them = mytype()
        for x in self.data_tables.keys():
            them._data[x] = self._data[x].copy()
        return them

    @property
    def excluded_channels(self):
        raise ValueError("Must be overridden")

    def binary_calls(self):
        return phenotype_calls()

    def phenotype_calls(self):
        # Default to just gating on mutually exclusive phenotypes
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
        

    @property
    def df(self):
        # a minimum useful dataframe
        #
        # <Project id> <Project name> optional (will only be on a project export)
        # <Sample id> <Sample name> optional (will only be on a sample export)
        # <Frame id> <Frame name>
        # <x> <y>
        # 
        #

        # get our region sizes
        region_sizes = self.get_data('regions').set_index('region_label')['region_size'].astype(int).to_dict()
        # get our cells
        temp1 = self.get_data('cells').drop(columns='phenotype_index').\
                       merge(self.get_data('regions'),
                             left_on='region_index',
                             right_index=True).drop(columns=['image_id','region_index','region_size'])
        temp1['regions'] = temp1.apply(lambda x: region_sizes,1)
        temp2 = self.scored_calls()
        if temp2  is not None:
            temp2 = temp2.apply(lambda x:
                dict(zip(
                    list(x.index),
                    list(x)
                 ))
            ,1).reset_index().rename(columns={0:'scored_calls'}).set_index('cell_index')
            temp1 = temp1.merge(temp2,left_index=True,right_index=True)
        else:
            temp1['scored_calls'] = np.nan
        temp3 = self.phenotype_calls().apply(lambda x:
                dict(zip(
                    list(x.index),
                    list(x)
                ))
            ,1).reset_index().rename(columns={0:'phenotype_calls'}).set_index('cell_index')
        
        temp1 = temp1.merge(temp3,left_index=True,right_index=True)
        #temp1['phenotypes_present'] = json.dumps(list(
        #        sorted([x for x in self.get_data('phenotypes')['phenotype_label'] if x is not np.nan])
        #    ))
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

        #temp5 = self.interaction_map().groupby('cell_index').\
        #    apply(lambda x: json.dumps(list(sorted(x['neighbor_cell_index'])))).reset_index().\
        #    rename(columns={0:'neighbor_cell_index'}).set_index('cell_index')


        # Get neighbor data .. may not be available for all cells
        neighbors = self.interaction_map().groupby('cell_index').\
            apply(lambda x:
                dict(zip(
                    x['neighbor_cell_index'].astype(int),x['pixel_count'].astype(int)
                ))
            ).reset_index().rename(columns={0:'neighbors'}).set_index('cell_index')
        edge_length = self.edge_map().reset_index().groupby('cell_index').count()[['x']].\
            rename(columns={'x':'edge_length'})
        edge_length['edge_length'] = edge_length['edge_length'].astype(int)

        cell_area = self.cell_map().reset_index().groupby('cell_index').count()[['x']].\
            rename(columns={'x':'cell_area'})
        cell_area['cell_area'] = cell_area['cell_area'].astype(int)
        temp5 = cell_area.merge(edge_length,left_index=True,right_index=True).merge(neighbors,left_index=True,right_index=True,how='left')
        temp5.loc[temp5['neighbors'].isna(),'neighbors'] = temp5.loc[temp5['neighbors'].isna(),'neighbors'].apply(lambda x: {}) # these are ones we actuall have measured

        temp1 = temp1.merge(temp5,left_index=True,right_index=True,how='left')
        temp1.loc[temp1['neighbors'].isna(),'neighbors'] = np.nan # These we were not able to measure


        temp1['frame_name'] = self.frame_name
        temp1['frame_id'] = self.id
        temp1  = temp1.reset_index()
        temp1 = temp1.sort_values('cell_index').reset_index(drop=True)
        temp1['sample_name'] = 'undefined'
        temp1['project_name'] = 'undefined'
        temp1['sample_id'] = 'undefined'
        temp1['project_id'] = 'undefined'
        def _get_phenotype(d):
            vals = [k for k,v in d.items() if v ==  1]
            return np.nan if len(vals) == 0 else vals[0]
        temp1['phenotype_label'] = temp1.apply(lambda x:
                  _get_phenotype(x['phenotype_calls'])
            ,1)
        return CellDataFrame(temp1)

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
        return self._id

    def create_cell_frame_class(self):
        return CellFrameGeneric()

    @property
    def frame_ids(self):
        return sorted(list(self._frames.keys()))

    @property
    def key(self):
        return self._key

    def get_frame(self,frame_id):
        return self._frames[frame_id]

    @property
    def df(self):
        output = []
        for frame_id in self.frame_ids:
            temp = self.get_frame(frame_id).df
            temp['sample_name'] = self.sample_name
            temp['sample_id'] = self.id
            output.append(temp)
        output = pd.concat(output).reset_index(drop=True)
        output.index.name = 'db_id'
        output['project_name'] = 'undefined'
        output['project_id'] = 'undefined'
        return CellDataFrame(pd.DataFrame(output))


    def to_hdf(self,h5file,location='',mode='w'):
        f = h5py.File(h5file,mode)
        #f.create_group(location+'/meta')
        #f.create_dataset(location+'/meta/id',data=self.id)
        #f.create_dataset(location+'/meta/sample_name',data=self.sample_name)
        dset = f.create_dataset(location+'/meta', (100,), dtype=h5py.special_dtype(vlen=str))
        dset.attrs['sample_name'] = self.sample_name
        dset.attrs['id'] = self._id
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

class CellProjectGeneric(object):
    def __init__(self,h5path,mode='r'):
        self._key = None
        self.h5path = h5path
        self.mode = mode
        if mode == 'w' or mode == 'r+':
            f = h5py.File(self.h5path,mode)
            f.create_group('/samples')
            dset = f.create_dataset('/meta', (100,), dtype=h5py.special_dtype(vlen=str))
            dset.attrs['project_name'] = np.nan
            dset.attrs['id'] = uuid4().hex
            f.close()

        #if mode == 'r' or mode == 'r+':
        #    #f = h5py.File(self.h5path,'r')
        #    #if 'info' in [x for x in f]: 
        #    #    self._key = pd.read_hdf(self.h5path,'info')
        #    #self.project_name = f['meta'].attrs['project_name']
        #    #self._id = f['meta'].attrs['id']
        #    #f.close()
        return

    @property
    def id(self):
        f = h5py.File(self.h5path,'r')
        name = f['meta'].attrs['id']
        f.close()
        return name

    @property 
    def project_name(self):
        f = h5py.File(self.h5path,'r')
        name = f['meta'].attrs['project_name']
        f.close()
        return name

    def set_project_name(self,name):
        f = h5py.File(self.h5path,'r+')
        #dset = f.create_dataset('/meta', (100,), dtype=h5py.special_dtype(vlen=str))
        f['meta'].attrs['project_name'] = name
        f.close()

    def set_id(self,name):
        f = h5py.File(self.h5path,'r+')
        #dset = f.create_dataset('/meta', (100,), dtype=h5py.special_dtype(vlen=str))
        f['meta'].attrs['id'] = name
        f.close()

    @property
    def df(self):
        output = []
        for sample_id in self.sample_ids:
            temp = self.get_sample(sample_id).df
            temp['project_name'] = self.project_name
            temp['project_id'] = self.id
            output.append(temp)
        output = pd.concat(output).reset_index(drop=True)
        output.index.name = 'db_id'
        return CellDataFrame(pd.DataFrame(output))

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

    #@property
    #def key(self):
    #    return self._key
    
    @property
    def sample_ids(self):
        return sorted(list(self.key['sample_id']))

    def get_sample(self,sample_id):
        sample = self.create_cell_sample_class()
        sample.read_hdf(self.h5path,'samples/'+sample_id)
        return sample

    @property
    def key(self):
        f = h5py.File(self.h5path,'r')
        val = False
        if 'info' in [x for x in f]: val = True
        f.close()
        return None if not val else pd.read_hdf(self.h5path,'info')
    
    def sample_iter(self):
        for sample_id in self.sample_ids: yield self.get_sample(sample_id)

    def frame_iter(self):
        for s in self.sample_iter():
            for frame_id in s.frame_ids:
                yield s.get_frame(frame_id)
