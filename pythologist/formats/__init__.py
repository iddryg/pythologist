""" These are classes to help deal with cell-level image data """
class CellImageDataGeneric(object):
    """ A generic CellImageData object
    """
    def __init__(self):
        self.data_tables = ['cell_locations',
                            'cell_phenotypes',
                            'cell_regions',
                            'cell_regions',
                            'cell_measurements',
                            'thresholds',
                            'measurement_features',
                            'measurement_channels',
                            'measurement_statistics',
                            'phenotypes',
                            'regions'
                           ]
        self.data = {}
        for x in self.data_tables: self.data[x] = None

    @property
    def thresholds(self):
        # Print the threhsolds
        return self.data['thresholds'].merge(self.data['measurement_statistics'],left_on='statistic_index',right_index=True).\
               merge(self.data['measurement_features'],left_on='feature_index',right_index=True).\
               merge(self.data['measurement_channels'],left_on='channel_index',right_index=True)

    def get_channels(self,all=False):
        if all: return self.data['measurement_channels']
        return self.data['measurement_channels'].loc[~self.data['measurement_channels']['channel_label'].isin(self.excluded_channels)]
    
    def get_raw(self,feature_label,statistic_label,all=False):
        stats = self.data['measurement_statistics'].reset_index()
        stats = stats.loc[stats['statistic_label']==statistic_label,'statistic_index'].iloc[0]
        feat = self.data['measurement_features'].reset_index()
        feat = feat.loc[feat['feature_label']==feature_label,'feature_index'].iloc[0]
        measure = self.data['cell_measurements']
        measure = measure.loc[(measure['statistic_index']==stats)&(measure['feature_index']==feat)]
        channels = self.data['measurement_channels']
        if not all: channels = channels.loc[~channels['channel_label'].isin(self.excluded_channels)]
        measure = measure.merge(channels,left_on='channel_index',right_index=True)
        return measure.reset_index().pivot(index='cell_index',columns='channel_label',values='value')

    def copy(self):
        # Do a deep copy of self
        mytype = type(self)
        them = mytype()
        for x in self.data_tables:
            them.data[x] = self.data[x].copy()
        return them

    @property
    def excluded_channels(self):
        raise ValueError("Must be overidden")

""" Hold a group of images from different samples """
class CellImageSetGeneric(object):
    def __init__(self):
        self.images = {}
        return