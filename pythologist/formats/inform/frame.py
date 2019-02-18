import os, re, json, sys
from collections import OrderedDict
import pandas as pd
import numpy as np
from pythologist.formats import CellFrameGeneric
from uuid import uuid4
from pythologist.formats.utilities import read_tiff_stack, map_image_ids, flood_fill, image_edges, watershed_image
import xml.etree.ElementTree as ET
from uuid import uuid4

_float_decimals = 6


class CellFrameInForm(CellFrameGeneric):
    """ Store data from a single image from an inForm export

        This is a CellFrame object that contains data and images from one image frame
    """
    def __init__(self):
        super().__init__()

        self._storage_type = np.float16

        ### Define extra InForm-specific data tables
        self.data_tables['thresholds'] = {'index':'gate_index',
                 'columns':['threshold_value','statistic_index',
                            'feature_index','channel_index',
                            'gate_label','region_index']}
        self.data_tables['mask_images'] = {'index':'db_id',
                 'columns':['mask_label','image_id']}
        for x in self.data_tables.keys():
            if x in self._data: continue
            self._data[x] = pd.DataFrame(columns=self.data_tables[x]['columns'])
            self._data[x].index.name = self.data_tables[x]['index']


    @property
    def excluded_channels(self):
        return ['Autofluorescence','Post-processing','DAPI']    

    @property
    def thresholds(self):
        # Print the threhsolds
        return self.get_data('thresholds').merge(self.get_data('measurement_statistics'),
                                                 left_on='statistic_index',
                                                 right_index=True).\
               merge(self.get_data('measurement_features'),
                     left_on='feature_index',
                     right_index=True).\
               merge(self.get_data('measurement_channels'),
                     left_on='channel_index',
                     right_index=True)

    def read_raw(self,
                 frame_name = None,
                 cell_seg_data_file=None,
                 cell_seg_data_summary_file=None,
                 score_data_file=None,
                 tissue_seg_data_file=None,
                 binary_seg_image_file=None,
                 component_image_file=None,
                 verbose=False,
                 channel_abbreviations=None,
                 require=True):
        self.frame_name = frame_name
        ### Read in the data for our object
        if verbose: sys.stderr.write("Reading text data.\n")
        self._read_data(cell_seg_data_file,
                   cell_seg_data_summary_file,
                   score_data_file,
                   tissue_seg_data_file,
                   verbose,
                   channel_abbreviations,require=require)
        if verbose: sys.stderr.write("Reading image data.\n")
        self._read_images(binary_seg_image_file,
                   component_image_file,
                   verbose=verbose,
                   require=require)
        return

    def default_raw(self):
        return self.get_raw(feature_label='Whole Cell',statistic_label='Mean')

    def binary_calls(self):
        # generate a table of gating calls with ncols = to the number of gates + phenotypes

        temp = self.phenotype_calls()
        if self.get_data('thresholds').shape[0] == 0:
            return temp.astype(np.int8)
        return temp.merge(self.scored_calls(),left_index=True,right_index=True).astype(np.int8)


    def binary_df(self):
        temp1 = self.phenotype_calls().stack().reset_index().\
            rename(columns={'level_1':'binary_phenotype',0:'score'})
        temp1.loc[temp1['score']==1,'score'] = '+'
        temp1.loc[temp1['score']==0,'score'] = '-'
        temp1['gated'] = 0
        temp2 = self._scored_gated_cells().stack().reset_index().\
            rename(columns={'gate_label':'binary_phenotype',0:'score'})
        temp2.loc[temp2['score']==1,'score'] = '+'
        temp2.loc[temp2['score']==0,'score'] = '-'
        temp2['gated'] = 1
        output = pd.concat([temp1,temp2])
        output.index.name = 'db_id'
        return output

    def scored_calls(self):
        d = self.get_data('thresholds').reset_index().\
            merge(self.get_data('cell_measurements').reset_index(),on=['statistic_index','feature_index','channel_index'])
        d['gate'] = d.apply(lambda x: x['value']>=x['threshold_value'],1)
        d = d.pivot(values='gate',index='cell_index',columns='gate_label').applymap(lambda x: 1 if x else 0)
        return d.astype(np.int8)
    

    def _read_data(self,
                        cell_seg_data_file=None,
                        cell_seg_data_summary_file=None,
                        score_data_file=None,
                        tissue_seg_data_file=None,
                        verbose=False,
                        channel_abbreviations=None,
                        require=True):
        """ Read in the image data from a inForm

        :param cell_seg_data_file:
        :type string:

        """
        _seg = pd.read_csv(cell_seg_data_file,"\t")
        if 'Tissue Category' not in _seg: _seg['Tissue Category'] = 'Any'

        ##########
        # Set the cells
        _cells = _seg.loc[:,['Cell ID','Cell X Position','Cell Y Position']].\
                              rename(columns={'Cell ID':'cell_index',
                                              'Cell X Position':'x',
                                              'Cell Y Position':'y'})
        _cells = _cells.applymap(int).set_index('cell_index')

        ###########
        # Set the cell phenotypes
        #    Try to read phenotypes from the summary file if its there because some may be zero counts and won't show up in the cell_seg file
        if 'Phenotype' in _seg:
            # Sometimes inform files won't have a Phenotype columns
            _phenotypes = _seg.loc[:,['Cell ID','Phenotype']]
        else:
            _phenotypes = _seg.loc[:,['Cell ID']]
            _phenotypes['Phenotype'] = np.nan
        _phenotypes = _phenotypes.rename(columns={'Cell ID':'cell_index','Phenotype':'phenotype_label'})
        _phenotypes_present = pd.Series(_phenotypes['phenotype_label'].unique()).tolist()
        if np.nan not in _phenotypes_present: _phenotypes_present = _phenotypes_present + [np.nan] 
        _phenotype_list = pd.DataFrame({'phenotype_label':_phenotypes_present})
        _phenotype_list.index.name = 'phenotype_index'
        _phenotype_list = _phenotype_list.reset_index()

        #if cell_seg_data_summary_file is not None:
        #     ############
        #     # Update the phenotypes table if a cell_seg_data_summary file is present
        #    if verbose: sys.stderr.write("Cell seg summary file is present so acquire phenotype list from it.\n")
        #    _segsum = pd.read_csv(cell_seg_data_summary_file,"\t")
        #    if 'Phenotype' not in _segsum.columns: 
        #        if verbose: sys.stderr.write("Missing phenotype column so set to NaN.\n")
        #        _segsum['Phenotype'] = np.nan

        #    _phenotypes_present = [x for x in sorted(_segsum['Phenotype'].unique().tolist()) if x != 'All']
        #    if np.nan not in _phenotypes_present: _phenotypes_present = _phenotypes_present + [np.nan] 
        #    _phenotype_list = pd.DataFrame({'phenotype_label':_phenotypes_present})
        #    _phenotype_list.index.name = 'phenotype_index'
        #    _phenotype_list = _phenotype_list.reset_index()

        _phenotypes = _phenotypes.merge(_phenotype_list,on='phenotype_label')
        _phenotype_list = _phenotype_list.set_index('phenotype_index')
        #Assign 'phenotypes' in a way that ensure we retain the pre-defined column structure
        self.set_data('phenotypes',_phenotype_list)
        if verbose: sys.stderr.write("Finished assigning phenotype list.\n")

        _phenotypes = _phenotypes.drop(columns=['phenotype_label']).applymap(int).set_index('cell_index')

        # Now we can add to cells our phenotype indecies
        _cells = _cells.merge(_phenotypes,left_index=True,right_index=True,how='left')


        ###########
        # Set the cell_regions
        _cell_regions = _seg[['Cell ID','Tissue Category']].copy().rename(columns={'Cell ID':'cell_index','Tissue Category':'region_label'})
        if tissue_seg_data_file:
            if verbose: sys.stderr.write("Tissue seg file is present.\n")
            _regions = pd.read_csv(tissue_seg_data_file,sep="\t")
            _regions = _regions[['Region ID','Tissue Category','Region Area (pixels)']].\
                rename(columns={'Region ID':'region_index','Tissue Category':'region_label','Region Area (pixels)':'region_size'}).set_index('region_index')
            # Set the image_id and region size to null for now
            _regions['image_id'] = np.nan # We don't have the image read in yet
            self.set_data('regions',_regions)
            #raise ValueError("Region summary not implemented")
        else:
            if verbose: sys.stderr.write("Tissue seg file is present.\n")
            _regions = pd.DataFrame({'region_label':_cell_regions['region_label'].unique()})
            _regions.index.name = 'region_index'
            _regions['region_size'] = np.nan # We don't have size available yet
            _regions['image_id'] = np.nan
            self.set_data('regions',_regions)
        _cell_regions = _cell_regions.merge(self.get_data('regions')[['region_label']].reset_index(),on='region_label')
        _cell_regions = _cell_regions.drop(columns=['region_label']).set_index('cell_index')

        # Now we can add to cells our region indecies
        _cells = _cells.merge(_cell_regions,left_index=True,right_index=True,how='left')

        # Assign 'cells' in a way that ensures we retain our pre-defined column structure. Should throw a warning if anything is wrong
        self.set_data('cells',_cells)
        if verbose: sys.stderr.write("Finished setting the cell list regions are set.\n")

        ###########
        # Get the intensity measurements - sets 'measurement_channels', 'measurement_statistics', 'measurement_features', and 'cell_measurements'
        self._parse_measurements(_seg,channel_abbreviations)  
        if verbose: sys.stderr.write("Finished setting the measurements.\n")
        ###########
        # Get the thresholds
        if score_data_file is not None: 
            self._parse_score_file(score_data_file)
            if verbose: sys.stderr.write("Finished reading score.\n")
        #self.set_data('binary_calls',self.binary_df())
        return

    def _parse_measurements(self,_seg,channel_abbreviations):   
        # Parse the cell seg pandas we've already read in to get the cell-level measurements, as well as what features we are measuring
        # Sets the 'measurement_channels', 'measurement_statistics', 'measurement_features', and 'cell_measurements'
        keepers = ['Cell ID']

        # Some older versions don't have tissue category
        if 'Entire Cell Area (pixels)' in _seg.columns: keepers.append('Entire Cell Area (pixels)')

        keepers2 = [x for x in _seg.columns if re.search('Entire Cell.*\s+\S+ \(Normalized Counts, Total Weighting\)$',x)]
        keepers3 = [x for x in _seg.columns if re.search('\s+\S+ \(Normalized Counts, Total Weighting\)$',x) and x not in keepers2]
        _intensity1 = []
        for cname in keepers2:
            m = re.match('Entire Cell\s+(.*) (Mean|Min|Max|Std Dev|Total) \(Normalized Counts, Total Weighting\)$',cname)
            stain = m.group(1)
            v = _seg[['Cell ID',cname]]
            v.columns = ['Cell ID','value']
            v = v.copy()
            for row in v.itertuples(index=False):
                _intensity1.append([row[0],stain,m.group(2),round(row[1],_float_decimals)])
        _intensity1 = pd.DataFrame(_intensity1,columns=['cell_index','channel_label','statistic_label','value'])
        _intensity1['feature_label'] = 'Whole Cell'

        _intensity2 = []
        #_intensity3 = []
        for cname in keepers3:
            if re.match('Entire Cell',cname): continue
            m = re.match('(\S+)\s+(.*) (Mean|Min|Max|Std Dev|Total) \(Normalized Counts, Total Weighting\)$',cname)
            compartment = m.group(1)
            stain = m.group(2)
            v = _seg[['Cell ID',cname,compartment+' Area (pixels)']]
            v.columns = ['Cell ID','value','value1']
            v = v.copy()
            for row in v.itertuples(index=False):
                _intensity2.append([row[0],stain,compartment,m.group(3),round(row[1],_float_decimals)])
                #_intensity3.append([row[0],'Post-processing',compartment,'Area (pixels)',round(row[2],_float_decimals)])

        _intensity2 = pd.DataFrame(_intensity2,columns=['cell_index','channel_label','feature_label','statistic_label','value'])
        #_intensity3 = pd.DataFrame(_intensity3,columns=['cell_index','channel_label','feature_label','statistic_label','value'])

        _intensities = [_intensity2,
                        #_intensity3,
                        _intensity1.loc[:,_intensity2.columns]]
        #if 'Entire Cell Area (pixels)' in _seg:
        #    _intensity4 = _seg[['Cell ID','Entire Cell Area (pixels)']].rename(columns={'Cell ID':'cell_index',
        #                                                                         'Entire Cell Area (pixels)':'value',
        #                                                                        })
        #    _intensity4['channel_label'] = 'Post-processing'
        #    _intensity4['feature_label'] = 'Whole Cell'
        #    _intensity4['statistic_label'] = 'Area (pixels)'
        #    _intensities += [_intensity4.loc[:,_intensity2.columns]]
        _intensity = pd.concat(_intensities)

        _measurement_channels = pd.DataFrame({'channel_label':_intensity['channel_label'].unique()})
        _measurement_channels.index.name = 'channel_index'
        _measurement_channels['channel_abbreviation'] = _measurement_channels['channel_label']
        if channel_abbreviations:
            _measurement_channels['channel_abbreviation'] = \
                _measurement_channels.apply(lambda x: x['channel_label'] if x['channel_label'] not in channel_abbreviations else channel_abbreviations[x['channel_label']],1)
        _measurement_channels['image_id'] = np.nan
        self.set_data('measurement_channels',_measurement_channels)

        _measurement_statistics = pd.DataFrame({'statistic_label':_intensity['statistic_label'].unique()})
        _measurement_statistics.index.name = 'statistic_index'
        self.set_data('measurement_statistics',_measurement_statistics)

        _measurement_features = pd.DataFrame({'feature_label':_intensity['feature_label'].unique()})
        _measurement_features.index.name = 'feature_index'
        self.set_data('measurement_features',_measurement_features)

        _cell_measurements = _intensity.merge(self.get_data('measurement_channels')[['channel_label','channel_abbreviation']].reset_index(),on='channel_label',how='left').\
                          merge(self.get_data('measurement_statistics').reset_index(),on='statistic_label',how='left').\
                          merge(self.get_data('measurement_features').reset_index(),on='feature_label',how='left').\
                          drop(columns=['channel_label','feature_label','statistic_label','channel_abbreviation'])
        _cell_measurements.index.name = 'measurement_index'
        _cell_measurements['cell_index'] = _cell_measurements['cell_index'].astype(np.uint32)
        self.set_data('cell_measurements',_cell_measurements)


    def _parse_score_file(self,score_data_file):
        # Sets the 'thresholds' table by parsing the score file
        _score_data = pd.read_csv(score_data_file,"\t")
        if 'Tissue Category' not in _score_data:
            raise ValueError('cannot read Tissue Category from '+str(score_file))
        _score_data.loc[_score_data['Tissue Category'].isna(),'Tissue Category'] = 'Any'
        ### We need to be careful how we parse this because there could be one or multiple stains in this file
        if 'Stain Component' in _score_data.columns:
            # We have the single stain case
            _score_data = _score_data[['Tissue Category','Cell Compartment','Stain Component','Positivity Threshold']].\
                      rename(columns={'Tissue Category':'region_label',
                                      'Cell Compartment':'feature_label',
                                      'Stain Component':'channel_label',
                                      'Positivity Threshold':'threshold_value'})
        elif 'First Stain Component' in _score_data.columns and 'Second Stain Component' in _score_data.columns:
            # lets break this into two tables and then merge them
            first_name = _score_data['First Stain Component'].iloc[0]
            second_name = _score_data['Second Stain Component'].iloc[0]
            table1 = _score_data[['Tissue Category','First Cell Compartment','First Stain Component',first_name+' Threshold']].\
                rename(columns ={
                    'Tissue Category':'region_label',
                    'First Cell Compartment':'feature_label',
                    'First Stain Component':'channel_label',
                    first_name+' Threshold':'threshold_value'
                    })
            table2 = _score_data[['Tissue Category','Second Cell Compartment','Second Stain Component',second_name+' Threshold']].\
                rename(columns ={
                    'Tissue Category':'region_label',
                    'Second Cell Compartment':'feature_label',
                    'Second Stain Component':'channel_label',
                    second_name+' Threshold':'threshold_value'
                    })
            _score_data = pd.concat([table1,table2]).reset_index(drop=True)
        else:
            # The above formats are the only known to exist in current exports
            raise ValueError("unknown score format")

        _score_data.index.name = 'gate_index'
        _score_data = _score_data.reset_index('gate_index')
        # We only want to read the 'Mean' statistic for thresholding
        _mystats = self.get_data('measurement_statistics')
        _score_data['statistic_index'] = _mystats[_mystats['statistic_label']=='Mean'].iloc[0].name 
        _thresholds = _score_data.merge(self.get_data('measurement_features').reset_index(),on='feature_label').\
                                  merge(self.get_data('measurement_channels')[['channel_label','channel_abbreviation']].reset_index(),on='channel_label').\
                                  merge(self.get_data('regions')[['region_label']].reset_index(),on='region_label').\
                                  drop(columns=['feature_label','channel_label','region_label'])
        # By default for inform name the gate after the channel abbreviation
        _thresholds['gate_label'] = _thresholds['channel_abbreviation']
        _thresholds = _thresholds.drop(columns=['channel_abbreviation'])
        _thresholds = _thresholds.set_index('gate_index')
        self.set_data('thresholds',_thresholds)

    ### Lets work with image files now
    def _read_images(self,binary_seg_image_file=None,component_image_file=None,verbose=False,require=True):
        # Start with the binary seg image file because if it has a processed image area,
        # that will be applied to all other masks and we can get that segmentation right away

        # Now we've read in whatever we've got fromt he binary seg image
        if verbose: sys.stderr.write("Reading component images.\n")
        if require or (not require and os.path.isfile(component_image_file)): 
            self._read_component_image(component_image_file)
        if verbose: sys.stderr.write("Finished reading component images.\n")

        if binary_seg_image_file is not None:
            if verbose: sys.stderr.write("Binary seg file present.\n")
            self._read_binary_seg_image(binary_seg_image_file)
            # if we have a ProcessedImage we can use that for an 'Any' region
            m = self.get_data('mask_images').set_index('mask_label')
            if 'ProcessRegionImage' in m.index:
                # we have a ProcessedImage
                #print('have a processedimage')
                self.set_processed_image_id(m.loc['ProcessRegionImage']['image_id'])
                self._images[self.processed_image_id] = self._images[self.processed_image_id].astype(np.int8)
            elif 'TissueClassMap' in m.index:
                # We can build a ProcessedImage from the TissueClassMap
                img = self._images[m.loc['TissueClassMap']['image_id']]
                self.set_processed_image_id(uuid4().hex)
                self._images[self.processed_image_id] = np.array(pd.DataFrame(img).applymap(lambda x: 0 if x==255 else 1)).astype(np.int8)
            segmentation_images = self.get_data('segmentation_images').set_index('segmentation_label')
            if 'Nucleus' in segmentation_images.index and \
               'Membrane' in segmentation_images.index:
                if verbose: sys.stderr.write("Making cell-map filled-in.\n")
                ## See if we are a legacy membrane map
                mem = self._images[self.get_data('segmentation_images').\
                          set_index('segmentation_label').loc['Membrane','image_id']]
                if len(pd.DataFrame(mem).unstack().reset_index()[0].unique()) == 2:
                    self._make_cell_map_legacy()
                else:
                    self._make_cell_map()
                if verbose: sys.stderr.write("Finished cell-map.\n")
                if verbose: sys.stderr.write("Making edge-map.\n")
                self._make_edge_map(verbose=verbose)
                if verbose: sys.stderr.write("Finished edge-map.\n")
                if verbose: sys.stderr.write("Set interaction map if appropriate")
                self.set_interaction_map(touch_distance=1)
            if verbose: sys.stderr.write("Finished reading seg file present.\n")

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

        # Now we can set the regions if we have them set intrinsically
        m = self.get_data('mask_images').set_index('mask_label')
        if 'TissueClassMap' in m.index:
            img = self._images[m.loc['TissueClassMap']['image_id']]
            regions = pd.DataFrame(img.astype(int)).stack().unique()
            regions = [x for x in regions if x != 255]
            region_key = []
            for region in regions:
                image_id = uuid4().hex
                region_key.append([region,image_id])
                self._images[image_id] = np.array(pd.DataFrame(img.astype(int)).applymap(lambda x: 1 if x==region else 0)).astype(np.int8)
            df = pd.DataFrame(region_key,columns=['region_index','image_id']).set_index('region_index')
            df['region_size'] = df.apply(lambda x:
                    self._images[x['image_id']].sum()
                ,1)
            temp = self.get_data('regions').drop(columns=['image_id','region_size']).merge(df,left_index=True,right_index=True,how='right')
            temp['region_size'] = temp['region_size'].astype(float)
            self.set_data('regions',temp)

        # If we don't have any regions set and all we have is 'Any' then we can just use the processed image
        _region = self.get_data('regions').query('region_label!="Any"').query('region_label!="any"')
        if _region.shape[0] ==0:
            if self.get_data('regions').shape[0] == 0: raise ValueError("Expected an 'Any' region")
            img = self._images[self._processed_image_id].copy()
            region_id = uuid4().hex
            self._images[region_id] = img
            df = pd.DataFrame(pd.Series({'region_index':0,'image_id':region_id,'region_size':img.sum()})).T.set_index('region_index')
            temp = self.get_data('regions').drop(columns=['image_id','region_size']).merge(df,left_index=True,right_index=True,how='right')
            temp['region_size'] = temp['region_size'].astype(float)
            self.set_data('regions',temp)

    def _read_component_image(self,filename):
        stack = read_tiff_stack(filename)
        channels = []
        for raw in stack:
            meta = raw['raw_meta']
            image_type, image_description = _parse_image_description(meta['ImageDescription'])
            if 'Name' not in image_description: continue
            channel_label = image_description['Name']
            image_id = uuid4().hex
            self._images[image_id] = raw['raw_image'].astype(self._storage_type)
            channels.append((channel_label,image_id))
        df = pd.DataFrame(channels,columns=['channel_label','image_id'])
        temp = self.get_data('measurement_channels').drop(columns=['image_id']).reset_index().merge(df,on='channel_label',how='left')
        self.set_data('measurement_channels',temp.set_index('channel_index'))
        return

    def _read_binary_seg_image(self,filename):
        stack = read_tiff_stack(filename)
        mask_names = []
        segmentation_names = []
        for raw in stack:
            meta = raw['raw_meta']
            image_type, image_description = _parse_image_description(meta['ImageDescription'])
            image_id = uuid4().hex
            if image_type == 'SegmentationImage':
                ### Handle if its a segmentation
                self._images[image_id] = raw['raw_image'].astype(int)
                segmentation_names.append([image_description['CompartmentType'],image_id])
            else:
                ### Otherwise it is a mask
                self._images[image_id] = raw['raw_image'].astype(int)
                mask_names.append([image_type,image_id])
        _mask_key = pd.DataFrame(mask_names,columns=['mask_label','image_id'])
        _mask_key.index.name = 'db_id'
        self.set_data('mask_images',_mask_key)
        _segmentation_key = pd.DataFrame(segmentation_names,columns=['segmentation_label','image_id'])
        _segmentation_key.index.name = 'db_id'
        self.set_data('segmentation_images',_segmentation_key)

    def _make_edge_map(self,verbose=False):
        #### Get the edges
        segmentation_images = self.get_data('segmentation_images').set_index('segmentation_label')
        cellid = segmentation_images.loc['cell_map','image_id']
        cm = self.get_image(cellid)
        memid = segmentation_images.loc['Membrane','image_id']
        mem = self.get_image(memid)
        em = image_edges(cm,verbose=verbose)
        em_id  = uuid4().hex
        self._images[em_id] = em.copy()
        increment  = self.get_data('segmentation_images').index.max()+1
        extra = pd.DataFrame(pd.Series(dict({'db_id':increment,
                                             'segmentation_label':'edge_map',
                                             'image_id':em_id}))).T
        extra = pd.concat([self.get_data('segmentation_images'),extra.set_index('db_id')])
        self.set_data('segmentation_images',extra)
        return em

    def _make_cell_map_legacy(self):
        #raise ValueError("legacy")


        segmentation_images = self.get_data('segmentation_images').set_index('segmentation_label')
        nucid = segmentation_images.loc['Nucleus','image_id']
        nuc = self.get_image(nucid)
        nmap = map_image_ids(nuc)

        memid = segmentation_images.loc['Membrane','image_id']
        mem = self.get_image(memid)
        mem = pd.DataFrame(mem).astype(float).applymap(lambda x: 9999999 if x > 0 else x)
        mem = np.array(mem)
        points = self.get_data('cells')[['x','y']]
        #points = points.loc[points.index.isin(nmap['id'])] # we may need this .. not sure
        output = np.zeros(mem.shape)
        for cell_index,v in points.iterrows():
            xi = v['x']
            yi = v['y']
            nums = flood_fill(mem,xi,yi,lambda x: x!=0,max_depth=1000,border_trim=1)
            if len(nums) >= 2000: continue
            for num in nums:
                if output[num[1]][num[0]] != 0: 
                    sys.stderr.write("Warning: skipping cell index overalap\n")
                    break 
                output[num[1]][num[0]] =  cell_index
        # Now fill out one point on all non-zeros into the zeros with watershed
        v = map_image_ids(output,remove_zero=False)
        zeros = v.loc[v['id']==0]
        zeros = list(zip(zeros['x'],zeros['y']))
        start = v.loc[v['id']!=0]
        start = list(zip(start['x'],start['y']))
        output = watershed_image(output,start,zeros,steps=1,border=1)

        cell_map_id  = uuid4().hex
        self._images[cell_map_id] = output.copy()
        increment  = self.get_data('segmentation_images').index.max()+1
        extra = pd.DataFrame(pd.Series(dict({'db_id':increment,
                                             'segmentation_label':'cell_map',
                                             'image_id':cell_map_id}))).T
        extra = pd.concat([self.get_data('segmentation_images'),extra.set_index('db_id')])
        self.set_data('segmentation_images',extra)

    def _make_cell_map(self):
        #### Get the cell map according to this ####
        #
        # Pre: Requires both a Nucleus and Membrane map
        # Post: Sets a 'cell_map' in the 'segmentation_images' 
        segmentation_images = self.get_data('segmentation_images').set_index('segmentation_label')
        nucid = segmentation_images.loc['Nucleus','image_id']
        memid = segmentation_images.loc['Membrane','image_id']
        nuc = self.get_image(nucid)
        mem = self.get_image(memid)


        nmap = map_image_ids(nuc)
        mmap = map_image_ids(mem)

        # get nuclear map coordinates that don't overlap the membrane
        overlap = nmap.rename(columns={'id':'nuc'}).\
                       merge(mmap.rename(columns={'id':'mem'}),on=['x','y'])
        overlap = set(overlap.apply(lambda x: (x['x'],x['y']),1))
        nmap['coord'] = nmap.apply(lambda x: (x['x'],x['y']),1)
        nmap = nmap.loc[~nmap['coord'].isin(overlap)]
        coord_x = nmap.groupby('id').apply(lambda x: sorted(list(x['x']))[int(len(x['x'])/2)]).reset_index().rename(columns={0:'x'}).reset_index()
        nmap = nmap.merge(coord_x,on=['id','x'])
        coord_y = nmap.groupby('id').apply(lambda x: sorted(list(x['y']))[int(len(x['y'])/2)]).reset_index().rename(columns={0:'y'}).reset_index()
        nmap = nmap.merge(coord_y,on=['id','y'])
        center = nmap.groupby('id').first()

        

        #print(self.get_data('cells').shape)
        #print(len(center))

        #center = self.get_data('cells')
        #center = center[['x','y']].copy()
        im = mem.copy()
        im2 = np.zeros(mem.shape).astype(int) #mem.copy()
        orig = pd.DataFrame(mem.copy())
        b1 = orig.iloc[0,:].sum()
        b2 = orig.iloc[:,0].sum()
        b3 = orig.iloc[orig.shape[0]-1,:].sum()
        b4 = orig.iloc[:,orig.shape[1]-1].sum()
        total = b1+b2+b3+b4
        #border_trim = 0
        #if total  == 0:
        #    border_trim = 2
        for cell_index in center.index:
            coord = (center.loc[cell_index]['x'],center.loc[cell_index]['y'])
            if im[coord[1]][coord[0]] != 0: 
                sys.stderr.write("Warning: skipping a cell center is exactly on the edge of a map.")
                continue
            num = flood_fill(im,coord[0],coord[1],lambda x: x!=0,max_depth=3000,border_trim=2)
            if len(num) >= 2000: continue 
            for v in num: 
                if im2[v[1]][v[0]] != 0 and im2[v[1]][v[0]] != cell_index: 
                    sys.stderr.write("Warning: skipping cell index overlap\n")
                    break 
                im2[v[1]][v[0]] = cell_index

        v = map_image_ids(im2,remove_zero=False)
        zeros = v.loc[v['id']==0]
        zeros = list(zip(zeros['x'],zeros['y']))
        start = v.loc[v['id']!=0]
        start = list(zip(start['x'],start['y']))

        im2 = watershed_image(im2,start,zeros,steps=1,border=1)

        c1 = map_image_ids(im2).reset_index().rename(columns={'id':'cell_index_1'})
        c2 = map_image_ids(im2).reset_index().rename(columns={'id':'cell_index_2'})
        overlap = c1.merge(c2,on=['x','y']).query('cell_index_1!=cell_index_2')
        if overlap.shape[0] > 0: raise ValueError("need to handle overlap")

        

        cell_map_id  = uuid4().hex
        self._images[cell_map_id] = im2.copy()
        increment  = self.get_data('segmentation_images').index.max()+1
        extra = pd.DataFrame(pd.Series(dict({'db_id':increment,
                                             'segmentation_label':'cell_map',
                                             'image_id':cell_map_id}))).T
        extra = pd.concat([self.get_data('segmentation_images'),extra.set_index('db_id')])
        self.set_data('segmentation_images',extra)


def _parse_image_description(metatext):
    root = ET.fromstring(metatext)
    d = dict([(child.tag,child.text) for child in root])
    return root.tag, d



