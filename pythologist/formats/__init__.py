import pandas as pd
""" These are classes to help deal with cell-level image data """
class CellImageDataGeneric(object):
    """ A generic CellImageData object
    """
    def __init__(self):
        # Define the column structure of all the tables.  
        #   Non-generic CellImageData could define additional data tables
        self.data_tables = {'cells':{'index':'cell_index',
                                     'columns':['x','y','phenotype_index','region_index']},
                            'cell_tags':{'index':'db_id',
                                         'columns':['tag_index','cell_index']},
                            'cell_measurements':{'index':'measurement_index',
                                                 'columns':['cell_index','statistic_index','feature_index','channel_index','value']},
                            'measurement_features':{'index':'feature_index',
                                                    'columns':['feature_label']},
                            'measurement_channels':{'index':'channel_index',
                                                    'columns':['channel_label','channel_abbreviation']},
                            'measurement_statistics':{'index':'statistic_index',
                                                      'columns':['statistic_label']},
                            'phenotypes':{'index':'phenotype_index',
                                          'columns':['phenotype_label']},
                            'regions':{'index':'region_index',
                                       'columns':['region_label']},
                            'tags':{'index':'tag_index',
                                    'columns':['tag_label']}
                           }
        self._data = {} # Do not acces directly. Use set_data_table and get_data_table to access.
        for x in self.data_tables.keys(): 
            self._data[x] = pd.DataFrame(columns=self.data_tables[x]['columns'])
            self._data[x].index.name = self.data_tables[x]['index']

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
    
    def get_raw(self,feature_label,statistic_label,region_label,all=False):
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
        return measure.reset_index().pivot(index='cell_index',columns='channel_label',values='value')

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

    @property
    def gated_cells(self):
        # Default to just gating on mutually exclusive phenotypes
        phenotypes = self.get_data('phenotypes')['phenotype_label'].dropna().tolist()
        temp = pd.DataFrame(index=self.get_data('cells').index,columns=phenotypes)
        temp = temp.fillna('-')
        temp = temp.merge(self.df[['phenotype_label']],left_index=True,right_index=True)
        for phenotype in phenotypes:
            temp.loc[temp['phenotype_label']==phenotype,phenotype]='+'
        return temp.drop(columns='phenotype_label')

    @property
    def df(self):
        # a dataframe that has phenotype and region info, but excludes all raw data
        return self.get_data('cells').merge(self.get_data('phenotypes'),left_on='phenotype_index',right_index=True,how='left').drop(columns='phenotype_index').\
                                      merge(self.get_data('regions'),left_on='region_index',right_index=True,how='left').drop(columns='region_index').sort_index()

    def complete_df(self):
        # a dataframe for every cell that has everything
        return

""" Hold a group of images from different samples """
class CellImageSetGeneric(object):
    def __init__(self):
        self.images = {}
        return