from pythologist_reader import CellFrameGeneric, CellSampleGeneric, CellProjectGeneric
from pythologist_image_utilities import read_tiff_stack, make_binary_image_array, image_edges, binary_image_dilation
import sys, json
from uuid import uuid4
import pandas as pd
import numpy as np

class CellProjectAnnotatedMIBI(CellProjectGeneric):
    def __init__(self,h5path,mode='r'):
        super().__init__(h5path,mode)
        return

    def create_cell_sample_class(self):
        return CellSampleAnnotatedMIBI()

    def read_json(self,input_data_json,run_parameters_json,verbose=False):
        # take a dictinary format input (compatible with json)
        project_name = input_data_json['project_name']
        self.project_name = project_name
        self.microns_per_pixel = run_parameters_json['microns_per_pixel']
        for sample_entry in input_data_json['samples']:
            csm = self.create_cell_sample_class()
            csm.read_json(sample_entry,run_parameters_json,verbose=verbose)
            csm.to_hdf(self.h5path,location='samples/'+csm.id,mode='a')

            current = self.key
            if current is None:
                current = pd.DataFrame([{'sample_id':csm.id,
                                     'sample_name':csm.sample_name}])
                current.index.name = 'db_id'
            else:
                iteration = max(current.index)+1
                addition = pd.DataFrame([{'db_id':iteration,
                                      'sample_id':csm.id,
                                      'sample_name':csm.sample_name}]).set_index('db_id')
                current = pd.concat([current,addition])
            current.to_hdf(self.h5path,'info',mode='r+',complib='zlib',complevel=9,format='table')
            if verbose: sys.stderr.write("Added Sample "+csm.id+"\n")

class CellSampleAnnotatedMIBI(CellSampleGeneric):
    def __init__(self):
        super().__init__()

    def create_cell_frame_class(self):
        return CellFrameAnnotatedMIBI()

    def read_json(self,input_data_json,run_parameters_json,verbose=False):
        # take a dictinary format input (compatible with json)
        sample_name = input_data_json['sample_name']
        frames = []
        for frame_entry in input_data_json['frames']:
            cfm = self.create_cell_frame_class()
            cfm.read_json(frame_entry,run_parameters_json,verbose=verbose)
            frame_id = cfm.id
            self._frames[frame_id]=cfm
            frames.append({'frame_id':frame_id,'frame_name':frame_entry['frame_name'],'frame_path':np.nan})
            if verbose: sys.stderr.write("Read Frame "+str(cfm.id)+"\n")
        self._key = pd.DataFrame(frames)
        self._key.index.name = 'db_id'
        self.sample_name = sample_name



class CellFrameAnnotatedMIBI(CellFrameGeneric):
    """ Store data from a single image from MIBI

        This is a CellFrame object that contains data and images from one image frame
    """
    def __init__(self):
        super().__init__()
        self.data_tables['mask_images'] = {'index':'db_id',
                 'columns':['mask_label','image_id']}
        for x in self.data_tables.keys():
            if x in self._data: continue
            self._data[x] = pd.DataFrame(columns=self.data_tables[x]['columns'])
            self._data[x].index.name = self.data_tables[x]['index']
    @property
    def excluded_channels(self):
    	return []
    def default_raw(self):
        return None

    def read_json(self,input_data_json,run_parameters_json,verbose=False):
    	# take a dictinary format input (compatible with json)
    	self.read_raw(
    		input_data_json['frame_name'],
    		input_data_json['annotations'],
    		input_data_json['mibi_cell_labels_tif_path'],
    		run_parameters_json['generate_processed_area_image'],
    		run_parameters_json['processed_area_image_steps'],
            run_parameters_json['channel_abbreviations'],
    		verbose
    	)

    def read_raw(self,
    	         frame_name=None,
                 annotations = None,
                 mibi_cell_labels_tif_path=None,
                 generate_processed_area_image=False,
                 processed_area_image_steps=50,
                 channel_abbreviations={},
                 verbose=False):
        self.verbose = verbose
        self.frame_name = frame_name
        if self.verbose: sys.stderr.write("Reading image data.\n")
        # Read images first because the tissue region tables need to be filled in so we can attribute a tissue region to data
        self._read_images(annotations,
                   mibi_cell_labels_tif_path,
                   generate_processed_area_image,
                   processed_area_image_steps,
                   channel_abbreviations)
        ### Read in the data for our object
        if self.verbose: sys.stderr.write("Reading text data.\n")
        self._read_data(annotations,
                   mibi_cell_labels_tif_path)
        self.set_interaction_map(touch_distance=1)
        return
    def _read_component_image(self,filename,channel_abbreviations):
        #stack = read_tiff_stack(filename)
        #channels = []
        #for index,v in enumerate(stack):
        #    img = v['raw_image']
        #    meta = v['raw_meta']
        #    image_description = json.loads(meta['ImageDescription'])
        #    channel_label = image_description['channel.target']
        #    image_id = uuid4().hex
        #    channels.append((channel_label,image_id))
        #    self._images[image_id] = img # saving image
        df = pd.DataFrame(channels,columns=['channel_label','image_id'])
        df['channel_abbreviation'] = df['channel_label'].apply(lambda x: x if x not in channel_abbreviations else channel_abbreviations[x])
        df.index.name = 'channel_index'
        self.set_data('measurement_channels',df)
        return 

    def _read_seg_image(self,mibi_cell_labels_tif_path,generate_processed_area_image,processed_area_image_steps):
    	# Do our segmentation first
        cell_labels = read_tiff_stack(mibi_cell_labels_tif_path)[0]['raw_image']
        image_id = uuid4().hex
        self._images[image_id] = cell_labels
        segmentation_names = [['Membrane',image_id]]
        _segmentation_key = pd.DataFrame(segmentation_names,columns=['segmentation_label','image_id'])
        _segmentation_key.index.name = 'db_id'
        self.set_data('segmentation_images',_segmentation_key)

        # make the cell map to be a copy of the membrane segmentation
        cell_map_id  = uuid4().hex
        self._images[cell_map_id] = cell_labels.copy()
        increment  = self.get_data('segmentation_images').index.max()+1
        extra = pd.DataFrame(pd.Series(dict({'db_id':increment,
                                             'segmentation_label':'cell_map',
                                             'image_id':cell_map_id}))).T
        extra = pd.concat([self.get_data('segmentation_images'),extra.set_index('db_id')])
        self.set_data('segmentation_images',extra)

        # make the edge map incrementally
        edge_map_id  = uuid4().hex
        self._images[edge_map_id] = image_edges(cell_labels)
        increment  = self.get_data('segmentation_images').index.max()+1
        extra = pd.DataFrame(pd.Series(dict({'db_id':increment,
                                             'segmentation_label':'edge_map',
                                             'image_id':edge_map_id}))).T
        extra = pd.concat([self.get_data('segmentation_images'),extra.set_index('db_id')])
        self.set_data('segmentation_images',extra)
      
        image_id = uuid4().hex
        mask_names = [['ProcessRegionImage',image_id]]
        processed = make_binary_image_array(np.ones(cell_labels.shape))
        self._images[image_id] = processed
        if self.verbose:
            sys.stderr.write("size of image: "+str(processed.sum().sum())+"\n")
        if generate_processed_area_image:
            if self.verbose: sys.stderr.write("Creating approximate processed_image_area by watershed.\n")
            processed = binary_image_dilation(make_binary_image_array(cell_labels),steps=processed_area_image_steps)
            self._images[image_id] = processed
        _mask_key = pd.DataFrame(mask_names,columns=['mask_label','image_id'])
        _mask_key.index.name = 'db_id'
        self.set_data('mask_images',_mask_key)
        self.set_processed_image_id(image_id)
        self._images[self.processed_image_id] = processed.astype(np.int8)
        if self.verbose:
            sys.stderr.write("size of processed image: "+str(processed.sum().sum())+"\n")

    def _read_images(self,annotations,mibi_cell_labels_tif_path,generate_processed_area_image,processed_area_image_steps,channel_abbreviations):
        # Start with the binary seg image file because if it has a processed image area,
        # that will be applied to all other masks and we can get that segmentation right away

        # Now we've read in whatever we've got fromt he binary seg image
        if self.verbose: sys.stderr.write("Reading component images.\n")
        df = pd.DataFrame((),columns=['channel_label','channel_abbreviation','image_id'])
        df.index.name = 'channel_index'
        self.set_data('measurement_channels',df)


        if self.verbose: sys.stderr.write("Finished reading component images.\n")
        self._read_seg_image(mibi_cell_labels_tif_path,generate_processed_area_image,processed_area_image_steps)


        _channel_key = self.get_data('measurement_channels')
        _channel_key_with_images = _channel_key[~_channel_key['image_id'].isna()]
        _channel_image_ids =  list(_channel_key.loc[~_channel_key['image_id'].isna(),'image_id'])

        _seg_key = self.get_data('segmentation_images')
        _seg_key_with_images = _seg_key[~_seg_key['image_id'].isna()]
        _seg_image_ids =  list(_seg_key.loc[~_seg_key['image_id'].isna(),'image_id'])
        _use_image_ids = _channel_image_ids+_seg_image_ids
        if self._processed_image_id is None and len(_use_image_ids)>0:
            # We have nothing so we assume the entire image is processed until we have some reason to update this
            if verbose: sys.stderr.write("No mask present so setting entire image area to be processed area.\n")
            dim = self._images[_use_image_ids[0]].shape                
            self._processed_image_id = uuid4().hex
            self._images[self._processed_image_id] = np.ones(dim,dtype=np.int8)

        if self._processed_image_id is None:

            raise ValueError("Nothing to set determine size of images")

        # ("Expected an 'Any' region")
        img = self._images[self._processed_image_id].copy()
        region_id = uuid4().hex
        self._images[region_id] = img
        df = pd.DataFrame(pd.Series({'region_index':0,'image_id':region_id,'region_size':img.sum()})).T.set_index('region_index')
        temp = self.get_data('regions').drop(columns=['image_id','region_size']).merge(df,left_index=True,right_index=True,how='right')
        temp['region_label'] = 'Any'
        temp['region_size'] = temp['region_size'].astype(float)
        self.set_data('regions',temp)

    def _read_data(self,
                   annotations,
                   mibi_cell_labels_tif_path):
        """ Read in the image data from MIBI using mibitracker

        :param cell_seg_data_file:
        :type string:

        """
        cell_labels = read_tiff_stack(mibi_cell_labels_tif_path)[0]['raw_image']
        if self.verbose: sys.stderr.write("Calling mibitracker cell total intensity scores\n")
        #_seg = mibi_segmentation.extract_cell_dataframe(cell_labels, mibi_tiff.read(mibi_component_tif_path), mode='total')
        #print(_seg)
        #if 'Tissue Category' not in _seg: _seg['Tissue Category'] = 'Any'

        _cells = pd.DataFrame(annotations).set_index('cell_index')


        ###########
        # Set dummy values for the cell phenotypes
        _phenotypes = _cells.copy().drop(columns=['x','y'])
        #_phenotypes['phenotype_label'] = np.nan
        _phenotypes = _phenotypes.loc[:,['phenotype_label']]
        _phenotypes_present = pd.Series(_phenotypes['phenotype_label'].unique()).tolist()
        _phenotypes_present = [np.nan] 
        _phenotype_list = pd.DataFrame({'phenotype_label':_cells['phenotype_label'].unique()})
        _phenotype_list.index.name = 'phenotype_index'
        _phenotype_list = _phenotype_list.reset_index()
        _phenotypes = _phenotypes.reset_index().merge(_phenotype_list,on='phenotype_label').set_index('cell_index')
        _phenotype_list = _phenotype_list.set_index('phenotype_index')

        #Assign 'phenotypes' in a way that ensure we retain the pre-defined column structure
        self.set_data('phenotypes',_phenotype_list)
        if self.verbose: sys.stderr.write("Finished assigning phenotype list.\n")

        _phenotypes = _phenotypes.drop(columns=['phenotype_label']).applymap(int)

        # Now we can add to cells our phenotype indecies
        _cells = _cells.merge(_phenotypes,left_index=True,right_index=True,how='left')

        ###########
        # Set the cell_regions

        _cell_regions = _cells[['x']].copy()
        _cell_regions['region_label'] = 'Any'
        _cell_regions = _cell_regions.drop(columns=['x'])


        _cell_regions = _cell_regions.reset_index().merge(self.get_data('regions')[['region_label']].reset_index(),on='region_label')
        _cell_regions = _cell_regions.drop(columns=['region_label']).set_index('cell_index')

        # Now we can add to cells our region indecies
        _cells = _cells.merge(_cell_regions,left_index=True,right_index=True,how='left').drop(columns=['phenotype_label'])

        # Assign 'cells' in a way that ensures we retain our pre-defined column structure. Should throw a warning if anything is wrong
        self.set_data('cells',_cells)
        if self.verbose: sys.stderr.write("Finished setting the cell list regions are set.\n")

        ###########
        # Get the intensity measurements - sets 'measurement_channels', 'measurement_statistics', 'measurement_features', and 'cell_measurements'

        #'cell_measurements':{'index':'measurement_index', 
        #                     'columns':['cell_index','statistic_index','feature_index','channel_index','value']},
        #'measurement_features':{'index':'feature_index',
        #                        'columns':['feature_label']},
        #'measurement_statistics':{'index':'statistic_index',
        #                          'columns':['statistic_label']},

        #_seg2 = _seg.reset_index().set_index(['label','area','x_centroid','y_centroid']).\
        #    stack().reset_index().rename(columns={'level_4':'marker',0:'Total'})
        #_seg2['feature_label'] = 'Total'

        #if self.verbose: sys.stderr.write("Calling mibitracker circular_sectors with 4 sectors\n")
        #_seg3 = mibi_segmentation.extract_cell_dataframe(cell_labels, mibi_tiff.read(mibi_component_tif_path), mode='circular_sectors', num_sectors=4).\
        #    reset_index().set_index(['label','area','x_centroid','y_centroid']).\
        #    stack().reset_index().rename(columns={'level_4':'marker',0:'Total'})
        #_seg3['feature_label'] = 'Circular Sectors 4'

        #if self.verbose: sys.stderr.write("Calling mibitracker circular_sectors with 8 sectors\n")
        #_seg4 = mibi_segmentation.extract_cell_dataframe(cell_labels, mibi_tiff.read(mibi_component_tif_path), mode='circular_sectors', num_sectors=8).\
        #    reset_index().set_index(['label','area','x_centroid','y_centroid']).\
        #    stack().reset_index().rename(columns={'level_4':'marker',0:'Total'})
        #_seg4['feature_label'] = 'Circular Sectors 8'

        #_segs = pd.concat([_seg2,_seg3,_seg4])
        #_segs['Mean'] = _segs.apply(lambda x: x['Total']/x['area'],1)
        #_segs = _segs.rename(columns={'label':'cell_index','marker':'channel_label'}).\
        #    drop(columns=['area','x_centroid','y_centroid']).set_index(['cell_index','channel_label','feature_label']).\
        #    stack().reset_index().rename(columns={0:'value','level_3':'statistic_label'})

        _measurement_statistics = pd.DataFrame([[0,'Total']],columns=['statistic_index','statistic_label']).\
                                     set_index('statistic_index')
        self.set_data('measurement_statistics',_measurement_statistics)

        _measurement_features = pd.DataFrame({'feature_label':['Total']})
        _measurement_features.index.name = 'feature_index'
        self.set_data('measurement_features',_measurement_features)


        _cell_measurements = pd.DataFrame((),columns=['cell_index','statistic_index', 'feature_index','channel_index', 'value'])
        _cell_measurements.index.name = 'measurement_index'
        self.set_data('cell_measurements',_cell_measurements)


        if self.verbose: sys.stderr.write("Finished setting the measurements.\n")
        ###########

        return
