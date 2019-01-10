import pandas as pd
import h5py, os
from uuid import uuid4
""" These are classes to help deal with cell-level image data """

class CellImageGeneric(object):
    """ A generic CellImageData object
    """
    def __init__(self):
        # Define the column structure of all the tables.  
        #   Non-generic CellImageData may define additional data tables
        self._processed_image_id = None
        self._images = {}                                  # Database of Images
        self._sample_name = None                                     # Sample name
        self._frame_name = None                                      # Individual image name from within a sample
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
        'regions':{'index':'region_index',
                   'columns':['region_label','region_size','image_id']},
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
    def sample_name(self):
        return self._sample_name
    @property
    def frame_name(self):
        return self._frame_name
    @property
    def sample_name(self):
       return self._sample_name
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

    @property
    def thresholds(self):
        raise ValueError('Override this to use it.')

    def get_channels(self,all=False):
        if all: return self.get_data('measurement_channels')
        d = self.get_data('measurement_channels')
        return d.loc[~d['channel_label'].isin(self.excluded_channels)]
    
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
        temp = temp.merge(self.df[['phenotype_label']],left_index=True,right_index=True)
        for phenotype in phenotypes:
            temp.loc[temp['phenotype_label']==phenotype,phenotype]=1
        return temp.drop(columns='phenotype_label').astype(np.int8)

    @property
    def df(self):
        # a dataframe that has phenotype and region info, but excludes all raw data
        return self.get_data('cells').merge(self.get_data('phenotypes'),left_on='phenotype_index',right_index=True,how='left').drop(columns='phenotype_index').\
                                      merge(self.get_data('regions'),left_on='region_index',right_index=True,how='left').drop(columns='region_index').sort_index()

    def complete_df(self):
        # a dataframe for every cell that has everything
        return

""" Hold a group of images from different samples """
class CellImageSet(object):
    # Store a collection of CellImage data objects
    # This object can be created 'w', read/written as a mutable object 'r+', or read-only 'r'
    def __init__(self,h5_db_filename,mode='r+'):
        if not mode in ['r','r+','w']: raise ValueError("Invalide mode selected "+mode)

        self._fn = h5_db_filename
        self._id = None
        self._mode = mode

        if 'id' in [x for x in h5py.File(self._fn,'r')] and self._mode != 'w':
            # We are doing some kind of reading
            self._id = pd.read_hdf(self._fn,'id').iloc[0,0]
        elif self._mode in ['r+','w']:
            # There is no id table so we will create a new database
            self._id = uuid4().hex
            df = pd.DataFrame(pd.Series({'id':self._id})).T
            df.to_hdf(self._fn,'id',mode='w',format='table')
        else:
            raise ValueError("CellImageSet has not been created yet, and is not in a redable format")
        return   

    @property
    def id(self):
        return self._id
    
    def get_data(self,table_name):
        return pd.read_hdf(self._fn,table_name)

    def add_data(self,cellimage):
        if self._mode == 'r': raise ValueError("cannot add in read-only mode")
 
        # Add a CellImage to the samples and return the index of this cellimage
        new_samples, cellimage_index = self._add_to_samples(cellimage)

        # update the cells and save the new index
        new_cells = self._add_to_cells(cellimage,cellimage_index)
        
        new_cell_measurements = self._add_to_cell_measurements(cellimage,cellimage_index,new_cells)


        # Write the changes
        new_samples.to_hdf(self._fn,'samples',mode='r+',format='table',complib='zlib',complevel=9)
        new_cells.to_hdf(self._fn,'cells',mode='r+',format='table',complib='zlib',complevel=9)
        new_cell_measurements.to_hdf(self._fn,'cell_measurements',mode='r+',format='table',complib='zlib',complevel=9)

    def _add_to_cell_measurements(self,cellimage,cellimage_index,new_cells):
        # the measurement index is not particularly important its fine if we reset it
        newdata = cellimage.get_data('cell_measurements')
        newdata['cellimage_index'] = cellimage_index
        if 'cell_measurements' in [x for x in h5py.File(self._fn,'r')]:
            #print(self.get_data('cells')[['cell_index','cellimage_index']].reset_index())
            olddata = self.get_data('cell_measurements').\
                merge(self.get_data('cells')[['cell_index','cellimage_index']],left_on='cell_dbid',right_index=True).\
                drop(columns='cell_dbid')
            newdata = pd.concat([olddata[newdata.columns],newdata]).reset_index(drop=True)
        newdata = newdata.merge(new_cells[['cell_index','cellimage_index']].reset_index(),on=['cell_index','cellimage_index']).\
            drop(columns=['cell_index','cellimage_index'])
        newdata.index.name = 'measurement_index'
        return newdata

    def _add_to_cells(self,cellimage,cellimage_index):
        newdata = cellimage.get_data('cells').reset_index()
        newdata['cellimage_index'] = cellimage_index
        if 'cells' in [x for x in h5py.File(self._fn,'r')]:
            olddata = self.get_data('cells')
            newdata = pd.concat([olddata[newdata.columns],newdata]).reset_index(drop=True)
        newdata.index.name='cell_dbid'
        return newdata

    def _add_binary_calls(self,cellimage,cellimage_index):
        newdata = cellimage.binary_calls().reset_index()
        newdata['cellimage_index'] = cellimage_index
        if 'binary_calls' in [x for x in h5py.File(self._fn,'r')]:
            olddata = self.get_data('binary_calls')
            newdata = pd.concat([olddata,newdata]).reset_index(drop=True)
        newdata = newdata.merge(self.get_data('cells')[['cell_index','cellimage_index']].reset_index(),on=['cell_index','cellimage_index']).\
            drop(columns=['cell_index','cellimage_index']).set_index('cell_dbid')
        newdata.to_hdf(self._fn,'binary_calls',mode='r+',format='table',complib='zlib',complevel=9)

    def _add_cellindex_to_table(self,cellimage,cellimage_index,table_name,indexname=None):
        # check if there is a regions table and create one if it doesn't already exist
        newdata = cellimage.get_data(table_name).reset_index()
        newdata['cellimage_index'] = cellimage_index
        if table_name in [x for x in h5py.File(self._fn,'r')]:
            regions = self.get_data(table_name)
            newdata = pd.concat([regions,newdata]).reset_index(drop=True)
        if indexname is not None:
            newdata.index.name = indexname
        newdata.to_hdf(self._fn,table_name,mode='r+',format='table',complib='zlib',complevel=9)

            

    def _add_to_samples(self,cellimage):
        # check if there is a sample table and create one if it doesn't already exist
        # return the cellimage index
        if not 'samples' in [x for x in h5py.File(self._fn,'r')]:
            # Table doesn't even exist
            samples_df = pd.DataFrame(pd.Series({'sample_name':cellimage.sample_name,
                                                'frame_name':cellimage.frame_name,
                                                'cellimage_id':cellimage.id})).T
            samples_df.index.name = 'cellimage_index'
            number = list(samples_df.index)[0]
        else:
            # We already have a samples table so add this data to our table
            samples_df = pd.read_hdf(self._fn,'samples')
            # check if this has already been added
            temp = samples_df[samples_df['cellimage_id']==cellimage.id]
            if temp.shape[0] > 0: raise ValueError("Error: cannot add the same cellimage twice")
            newdata = pd.DataFrame(pd.Series({'sample_name':cellimage.sample_name,
                                              'frame_name':cellimage.frame_name,
                                              'cellimage_id':cellimage.id})).T
            newdata.index.name = 'cellimage_index'
            number = samples_df.index.max()+1
            newdata.index = [number]
            samples_df = pd.concat([samples_df,newdata])
            samples_df.index.name = 'cellimage_index'
        return samples_df,number


        
        