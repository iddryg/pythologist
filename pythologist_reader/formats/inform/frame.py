import os, re, json, sys
from collections import OrderedDict
import pandas as pd
import numpy as np
from pythologist_reader import CellFrameGeneric
from uuid import uuid4
from pythologist_image_utilities import read_tiff_stack, map_image_ids, \
                                        image_edges, watershed_image, \
                                        median_id_coordinates
from skimage.segmentation import flood_fill, flood
import xml.etree.ElementTree as ET
from uuid import uuid4
import xmltodict

_float_decimals = 6


class CellFrameInForm(CellFrameGeneric):
    """ Store data from a single image from an inForm export

        This is a CellFrame object that contains data and images from one image frame
    """
    def __init__(self):
        super().__init__()

        self._storage_type = None

        ### Define extra InForm-specific data tables

        # This one is not quite right ... the 'region_index' is not currently implemented, and is not used for applying threhsolds,
        # so in the end we should set this to NaN.  I would remove it but it would break some backwards compatibility.
        # we never have gotten a project where they have threhsolded differently on different regions, and this isn't something we
        # should probably support unless we have a really good reason too
        self.data_tables['thresholds'] = {'index':'gate_index',
                 'columns':['threshold_value','statistic_index',
                            'feature_index','channel_index',
                            'feature_label','region_index']}
        self.data_tables['mask_images'] = {'index':'db_id',
                 'columns':['mask_label','image_id']}
        for x in self.data_tables.keys():
            if x in self._data: continue
            self._data[x] = pd.DataFrame(columns=self.data_tables[x]['columns'])
            self._data[x].index.name = self.data_tables[x]['index']


    @property
    def excluded_channels(self):
        return ['Autofluorescence','Post-processing','DAPI']    

    #@property
    #def thresholds(self):
    #    # Print the threhsolds
    #    return self.get_data('thresholds').merge(self.get_data('measurement_statistics'),
    #                                             left_on='statistic_index',
    #                                             right_index=True).\
    #           merge(self.get_data('measurement_features'),
    #                 left_on='feature_index',
    #                 right_index=True).\
    #           merge(self.get_data('measurement_channels'),
    #                 left_on='channel_index',
    #                 right_index=True)

    def read_raw(self,
                 frame_name = None,
                 cell_seg_data_file=None,
                 score_data_file=None,
                 binary_seg_image_file=None,
                 component_image_file=None,
                 verbose=False,
                 inform_analysis_dict=None,
                 require_component=True,
                 require_score=True,
                 skip_segmentation_processing=False):
        self.frame_name = frame_name
        if verbose: sys.stderr.write("Reading image data.\n")


        def _parse_threshold(inform_analysis_dict):
            # return a simpler mapping of inform labels to new labels for anything we keep
            _output = {}
            for _d in inform_analysis_dict['threshold_features']:
                if 'keep' not in _d: 
                    _d['keep'] = True
                if _d['keep'] is False: continue
                _output[_d['inform_channel_label']] = _d['label']
            return _output
        def _parse_mutually_exclusive(inform_analysis_dict):
            # return a tuple keyed dictionary
            _output = {}
            for _strategy in inform_analysis_dict['mutually_exclusive_phenotype_strategies']:
                # iterating over each strategy
                _strategy_tuple = tuple(sorted([x['assigned_label'] for x in _strategy]))
                _output[_strategy_tuple] = {}
                for _pheno in _strategy:
                    if 'keep' not in _pheno:
                        _pheno['keep'] = True
                    if _pheno['keep'] is False: continue
                    _output[_strategy_tuple][_pheno['assigned_label']] = \
                        _pheno['assigned_label'] if 'label' not in _pheno else _pheno['label']
            return _output
       

        threshold_analysis = _parse_threshold(inform_analysis_dict)
        mutually_exclusive_analysis = _parse_mutually_exclusive(inform_analysis_dict)


        # Read images first because the tissue region tables need to be filled in so we can attribute a tissue region to data
        self._read_images(binary_seg_image_file,
                   component_image_file,
                   verbose=verbose,
                   require_component=require_component,
                   skip_segmentation_processing=skip_segmentation_processing)
        ### Read in the data for our object
        if verbose: sys.stderr.write("Reading text data.\n")
        self._read_data(cell_seg_data_file,
                   score_data_file,
                   threshold_analysis,
                   mutually_exclusive_analysis,
                   verbose,
                   require_component=require_component,
                   require_score=require_score,
                   skip_segmentation_processing=skip_segmentation_processing)
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
        if self.get_data('thresholds').shape[0] == 0: return None
        d = self.get_data('thresholds').reset_index().\
            merge(self.get_data('cell_measurements').reset_index(),on=['statistic_index','feature_index','channel_index'])
        d['gate'] = d.apply(lambda x: x['value']>=x['threshold_value'],1)
        d = d.pivot(values='gate',index='cell_index',columns='gate_label').applymap(lambda x: 1 if x else 0)
        return d.astype(np.int8)
    

    def _read_data(self,
                        cell_seg_data_file,
                        score_data_file,
                        threshold_analysis,
                        mutually_exclusive_analysis,
                        verbose=False,
                        require_component=True,
                        require_score=True,
                        skip_segmentation_processing=False):
        """ Read in the image data from a inForm

        :param cell_seg_data_file:
        :type string:

        """


        _seg = pd.read_csv(cell_seg_data_file,sep="\t")
        if 'Tissue Category' not in _seg: _seg['Tissue Category'] = 'Any'

        ##########
        # Set the cells
        _cells = _seg.loc[:,['Cell ID','Cell X Position','Cell Y Position']].\
                              rename(columns={'Cell ID':'cell_index',
                                              'Cell X Position':'x',
                                              'Cell Y Position':'y'})
        _cells = _cells.applymap(int).set_index('cell_index')

        
        pheno_columns = [x for x in _seg.columns if 'Phenotype-' in x]
        if len(pheno_columns)==0:
            pheno_columns = ['Phenotype']

        logged_phenotypes = set()
        me_features = []
        me_feature_definition = []
        for pc in pheno_columns:
            pstr = re.match('Phenotype-(.*)$',pc).group(1)
            _ptuple = tuple(sorted(pstr.split(', ')))
            _sub = _seg.loc[:,['Cell ID',pc]].copy().\
                rename(columns={'Cell ID':'cell_index',pc:'Phenotype'}).\
                dropna(subset=['Phenotype'])
            # we are only keeping a subset within these
            if _ptuple not in mutually_exclusive_analysis:
                raise ValueError("Missing expected Phenotype analysis column "+str(_ptup))
            for _assigned, _label in mutually_exclusive_analysis[_ptuple].items():
                me_feature_definition.append([_label,'+',1])
                me_feature_definition.append([_label,'-',0])
                _s1 = _sub.copy()
                _s1['feature_value'] = _s1.apply(lambda x: 1 if x['Phenotype'] == _assigned else 0,1)
                _s1['feature_label'] = _label
                _s1 = _s1.drop(columns=['Phenotype'])
                if _label in logged_phenotypes:
                    raise ValueError("Error, of a repeated phenotype label "+str(_label))
                logged_phenotypes.add(_label)
                me_features.append(_s1)
        me_features = pd.concat(me_features).reset_index(drop=True)


        ###########
        # Set the cell_regions
        _cell_regions = _seg[['Cell ID','Tissue Category']].copy().rename(columns={'Cell ID':'cell_index','Tissue Category':'region_label'})


        _cell_regions = _cell_regions.merge(self.get_data('regions')[['region_label']].reset_index(),on='region_label')
        _cell_regions = _cell_regions.drop(columns=['region_label']).set_index('cell_index')

        # Now we can add to cells our region indecies
        _cells = _cells.merge(_cell_regions,left_index=True,right_index=True,how='left')

        # Assign 'cells' in a way that ensures we retain our pre-defined column structure. Should throw a warning if anything is wrong
        self.set_data('cells',_cells)
        if verbose: sys.stderr.write("Finished setting the cell list regions are set.\n")

        ###########
        # Get the intensity measurements - sets 'measurement_channels', 'measurement_statistics', 'measurement_features', and 'cell_measurements'
        self._parse_measurements(_seg,threshold_analysis)  
        if verbose: sys.stderr.write("Finished setting the measurements.\n")
        ###########
        # Get the thresholds
        if score_data_file is None and len(threshold_analysis.keys())>0:
            raise ValueError("Expecting threshold data but no score data file provided")

        extracted_threshold = []
        if score_data_file is not None: 
            _thresholds = preliminary_threshold_read(score_data_file, 
                                                 self.get_data('measurement_statistics'), 
                                                 self.get_data('measurement_features'), 
                                                 self.get_data('measurement_channels'), 
                                                 self.get_data('regions'))
            if verbose: sys.stderr.write("Finished reading score.\n")
            self.set_data('thresholds',_thresholds)

            ms = self.get_data('measurement_statistics')
            mf = self.get_data('measurement_features')
            mc = self.get_data('measurement_channels')
            cm = self.get_data('cell_measurements')
            cells = self.get_data('cells')
            regions = self.get_data('regions')

            _t = cells.merge(regions,left_on='region_index',right_index=True).\
                       drop(columns=['x','y','region_size','image_id']).\
                       merge(cm,left_index=True,right_on=['cell_index']).\
                       merge(_thresholds,on=['statistic_index','feature_index','channel_index','region_index'])
            _t['feature_value'] = _t.apply(lambda x: 1 if x['value']>=x['threshold_value'] else 0,1)
            _t = _t.loc[:,['cell_index','feature_label','feature_value']]
            _t = _t.loc[_t['feature_label'].isin([x for x in threshold_analysis.values()]),:]
            _flabs = _t['feature_label'].unique()
            for _k,_v in threshold_analysis.items():
                if _v not in _flabs: raise ValueError("Missing threshold feature "+str(_v))
            t_features = _t.copy()

        t_feature_definition = []
        for _k,_v in threshold_analysis.items():
            t_feature_definition.append([_v,'+',1])
            t_feature_definition.append([_v,'-',0])
        feature_definition = pd.DataFrame(me_feature_definition+t_feature_definition,
                                          columns=['feature_label','feature_value_label','feature_value']).\
                                          reset_index(drop=True)
        feature_definition.index.name = 'feature_index'
        features = pd.concat([me_features,t_features]).reset_index(drop=True).\
            merge(feature_definition.reset_index(),on=['feature_label','feature_value'])[['cell_index','feature_index']]
        features.index.name = 'db_id'

        self.set_data('cell_features',features)
        self.set_data('features',feature_definition)

        return

    def _parse_measurements(self,_seg,threshold_analysis):   
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

        _intensity = pd.concat(_intensities)

        _measurement_channels = pd.DataFrame({'channel_label':_intensity['channel_label'].unique()})
        _measurement_channels.index.name = 'channel_index'
        _measurement_channels['channel_abbreviation'] = _measurement_channels['channel_label']
        if threshold_analysis:
            _measurement_channels['channel_abbreviation'] = \
                _measurement_channels.apply(lambda x: x['channel_label'] if x['channel_label'] not in threshold_analysis else threshold_analysis[x['channel_label']],1)
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

        
        return _thresholds

    ### Lets work with image files now
    def _read_images(self,binary_seg_image_file,
                          component_image_file=None,
                          verbose=False,
                          require_component=True,
                          skip_segmentation_processing=False):
        # Start with the binary seg image file because if it has a processed image area,
        # that will be applied to all other masks and we can get that segmentation right away

        # Now we've read in whatever we've got fromt he binary seg image
        if verbose: sys.stderr.write("Reading component images.\n")
        if require_component or (not require_component and component_image_file and os.path.isfile(component_image_file)): 
            self._read_component_image(component_image_file)
        if verbose: sys.stderr.write("Finished reading component images.\n")

        if binary_seg_image_file is not None:
            if verbose: sys.stderr.write("Binary seg file present.\n")
            self._read_binary_seg_image(binary_seg_image_file)
            # if we have a ProcessedImage we can use that for an 'Any' region
            m = self.get_data('mask_images').set_index('mask_label')
            ### Set a procssed image area based on available layers in the binary seg image file
            if 'ProcessRegionImage' in m.index:
                # we have a ProcessedImage
                self.set_processed_image_id(m.loc['ProcessRegionImage']['image_id'])
                self._images[self.processed_image_id] = self._images[self.processed_image_id].astype(np.int8)
            elif 'TissueClassMap' in m.index:
                # Alternatively we can build a ProcessedImage from the TissueClassMap
                img = self._images[m.loc['TissueClassMap']['image_id']]
                self.set_processed_image_id(uuid4().hex)
                self._images[self.processed_image_id] = np.array(pd.DataFrame(img).applymap(lambda x: 0 if x==255 else 1)).astype(np.int8)

            # If we don't have regions already, make a regions


            segmentation_images = self.get_data('segmentation_images').set_index('segmentation_label')
            if 'Nucleus' in segmentation_images.index and \
               'Membrane' in segmentation_images.index and not skip_segmentation_processing:
                if verbose: sys.stderr.write("Making cell-map filled-in.\n")
                ## See if we are a legacy membrane map
                mem = self._images[self.get_data('segmentation_images').\
                          set_index('segmentation_label').loc['Membrane','image_id']]
                color_count = len(pd.DataFrame(mem).unstack().reset_index()[0].unique())
                if color_count < 10:
                    if verbose: sys.stderr.write("Only found "+str(color_count)+" colors on the membrane looks like legacy\n")
                    self._make_cell_map_legacy()
                else:
                    self._make_cell_map()
                if verbose: sys.stderr.write("Finished cell-map.\n")
                if verbose: sys.stderr.write("Making edge-map.\n")
                self._make_edge_map(verbose=verbose)
                if verbose: sys.stderr.write("Finished edge-map.\n")
                if verbose: sys.stderr.write("Set interaction map if appropriate\n")
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

        # If we don't have any regions set and all we have is 'Any' then we can just use the processed image
        _region = self.get_data('regions') #.query('region_label!="Any"').query('region_label!="any"')
        if _region.shape[0] ==0:
            #if self.get_data('regions').shape[0] == 0: raise ValueError("Expected an 'Any' region")
            img = self._images[self._processed_image_id].copy()
            region_id = uuid4().hex
            self._images[region_id] = img
            df = pd.DataFrame(pd.Series({'region_index':0,'image_id':region_id,'region_size':img.sum()})).T.set_index('region_index')
            temp = self.get_data('regions').drop(columns=['image_id','region_size']).merge(df,left_index=True,right_index=True,how='right')
            temp['region_label'] = 'Any'
            temp['region_size'] = temp['region_size'].astype(float)
            self.set_data('regions',temp)

    def _read_component_image(self,filename):
        stack = read_tiff_stack(filename)
        channels = []
        for raw in stack:
            meta = raw['raw_meta']
            image_type, image_description = self._parse_image_description(meta['ImageDescription'])
            #image_type, image_description = self._parse_image_description(meta['ImageDescription'])
            if 'ImageType' not in image_description: continue
            if image_description['ImageType'] == 'ReducedResolution': continue
            if 'Name' not in image_description: continue
            channel_label = image_description['Name']
            image_id = uuid4().hex
            if self._storage_type is not None:
                self._images[image_id] = raw['raw_image'].astype(self._storage_type)
            else:
                self._images[image_id] = raw['raw_image']
            channels.append((channel_label,image_id))
        df = pd.DataFrame(channels,columns=['channel_label','image_id'])
        temp = self.get_data('measurement_channels').drop(columns=['image_id']).reset_index().merge(df,on='channel_label',how='left')
        self.set_data('measurement_channels',temp.set_index('channel_index'))
        return

    def _parse_image_description(self,metatext):
        #root = ET.fromstring(metatext.decode('utf-8'))
        d = xmltodict.parse(metatext)
        if len(list(d.keys())) > 1: raise ValueError("Unexpected XML format with multiple root tags")
        root_tag = list(d.keys())[0]
        #d = dict([(child.tag,child.text) for child in root])
        #return root.tag, d
        return root_tag, d[root_tag]

    def _read_binary_seg_image(self,filename):
        stack = read_tiff_stack(filename)
        mask_names = []
        segmentation_names = []
        for raw in stack:
            meta = raw['raw_meta']
            image_type, image_description = self._parse_image_description(meta['ImageDescription'])

            image_id = uuid4().hex
            if image_type == 'SegmentationImage':
                ### Handle if its a segmentation
                self._images[image_id] = raw['raw_image'].astype(int)
                segmentation_names.append([image_description['CompartmentType'],image_id])
            else:
                ### Otherwise it is a mask
                self._images[image_id] = raw['raw_image'].astype(int)
                mask_names.append([image_type,image_id])
                ### If we have a Tissue Class Map we should process it into regions
                if image_type == 'TissueClassMap':
                    # Process the Tissue Class Map
                    self._process_tissue_class_map(image_description,raw['raw_image'].astype(int))


        _mask_key = pd.DataFrame(mask_names,columns=['mask_label','image_id'])
        _mask_key.index.name = 'db_id'
        self.set_data('mask_images',_mask_key)
        _segmentation_key = pd.DataFrame(segmentation_names,columns=['segmentation_label','image_id'])
        _segmentation_key.index.name = 'db_id'
        self.set_data('segmentation_images',_segmentation_key)

    def _process_tissue_class_map(self,image_description,img):
        # Now we can set the regions if we have them set intrinsically

        regions = pd.DataFrame(img.astype(int)).stack().unique()
        regions = [x for x in regions if x != 255]
        region_key = []

        for region in regions:
            image_id = uuid4().hex

            # if value of key 'Entry' is list, pass
            # if value of key 'Entry' is dictionary create a dictionary to index region
            if isinstance(image_description['Entry'], list):
                region_label = image_description['Entry'][region - 1]['Name']
            else:
                entry_list = dict(image_description['Entry'].items())
                image_description['Entry'] = entry_list
                region_label = image_description['Entry'].get("Name")

            region_key.append([region,region_label,image_id])
            self._images[image_id] = np.array(pd.DataFrame(img.astype(int)).applymap(lambda x: 1 if x==region else 0)).astype(np.int8)
        df = pd.DataFrame(region_key,columns=['region_index','region_label','image_id']).set_index('region_index')
        df['region_size'] = df.apply(lambda x:
            self._images[x['image_id']].sum()
        ,1)

        self.set_data('regions',df[['region_label','image_id','region_size']])








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
        from pythologist_image_utilities import flood_fill
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
        output = np.zeros(mem.shape).astype(int)
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
        output = watershed_image(output,start,zeros,steps=1,border=1).astype(int)
        # Now we need to clean up the image
        # Try to identify cells that are overlapping the processed image
        if self.processed_image_id is not None:
            ci = map_image_ids(output,remove_zero=False)
            pi = map_image_ids(self.processed_image,remove_zero=False)
            mi = ci.merge(pi,on=['x','y'])
            bad = mi.loc[(mi['id_y']==0)&(mi['id_x']!=0),'id_x'].unique() # find the bad
            #if self.verbose: sys.stderr.write("Removing "+str(bad.shape)+" bad points")
            mi.loc[mi['id_x'].isin(bad),'id_x'] = 0 # set the bad to zero
            output = np.array(mi.pivot(columns='x',index='y',values='id_x'))


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
        mids = map_image_ids(mem)
        coords = list(zip(mids['x'],mids['y']))
        center =  median_id_coordinates(nuc,coords)
        im = mem.copy()
        im2 = mem.copy()
        orig = pd.DataFrame(mem.copy())

        for i,cell_index in enumerate(center.index):
            coord = (center.loc[cell_index]['x'],center.loc[cell_index]['y'])
            mask = flood(im2,(coord[1],coord[0]),connectivity=1,tolerance=0)
            if mask.sum().sum() >= 2000: continue
            im2[mask] = cell_index

        v = map_image_ids(im2,remove_zero=False)
        zeros = v.loc[v['id']==0]
        zeros = list(zip(zeros['x'],zeros['y']))
        start = v.loc[v['id']!=0]
        start = list(zip(start['x'],start['y']))

        c1 = map_image_ids(im2).reset_index().rename(columns={'id':'cell_index_1'})
        c2 = map_image_ids(im2).reset_index().rename(columns={'id':'cell_index_2'})
        overlap = c1.merge(c2,on=['x','y']).query('cell_index_1!=cell_index_2')
        if overlap.shape[0] > 0: raise ValueError("need to handle overlap")

        
        #print("DONE FILLING IN")
        cell_map_id  = uuid4().hex
        self._images[cell_map_id] = im2.copy()
        increment  = self.get_data('segmentation_images').index.max()+1
        extra = pd.DataFrame(pd.Series(dict({'db_id':increment,
                                     'segmentation_label':'cell_map',
                                     'image_id':cell_map_id}))).T
        extra = pd.concat([self.get_data('segmentation_images'),extra.set_index('db_id')])
        self.set_data('segmentation_images',extra)

def preliminary_threshold_read(score_data_file, measurement_statistics, measurement_features, measurement_channels, regions):
    # We create an exportable function for this feature because we want to be able to get a thresholds table 
    # for alternate exports without reading in everything.

    # Sets the 'thresholds' table by parsing the score file
    _score_data = pd.read_csv(score_data_file,sep="\t")
    # Now we know there are two possible scoring schemes... a "Positivity Threshold" for binary type,
    # or a series of Thresholds for ordinal types
    # lets see which of our types we have
        
    is_ordinal = False
    if 'Threshold 0/1+' in _score_data.columns:
        is_ordinal = True

    if 'Tissue Category' not in _score_data:
        raise ValueError('cannot read Tissue Category from '+str(score_file))
    _score_data.loc[_score_data['Tissue Category'].isna(),'Tissue Category'] = 'Any'


    def _parse_binary(_score_data,measurement_statistics,measurement_features,measurement_channels,regions):
        _score_data=_score_data.copy()
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

        _mystats = measurement_statistics
        _score_data['statistic_index'] = _mystats[_mystats['statistic_label']=='Mean'].iloc[0].name 
        _thresholds = _score_data.merge(measurement_features.reset_index(),on='feature_label').\
                                  merge(measurement_channels[['channel_label','channel_abbreviation']].reset_index(),on='channel_label').\
                                  merge(regions[['region_label']].reset_index(),on='region_label').\
                                  drop(columns=['feature_label','channel_label','region_label'])
        # By default for inform name the gate after the channel abbreviation
        _thresholds['feature_label'] = _thresholds['channel_abbreviation']
        _thresholds = _thresholds.drop(columns=['channel_abbreviation'])
        _thresholds = _thresholds.set_index('gate_index')


        return _thresholds

    def _parse_ordinal(_score_data,measurement_statistics,measurement_features,measurement_channels,regions):
        _score_data = _score_data.copy()
        # Identify the "Threshold" columns
        mre = re.compile('Threshold (\d+)\+?/(\d+)\+?')
        threshold_columns = [(x,mre.match(x).group(1)+'/'+mre.match(x).group(2)) for x in list(_score_data.columns) if mre.match(x)]
        threshold_dict = dict(threshold_columns)
        #print(threshold_columns)
        static_columns = ['Tissue Category','Cell Compartment','Stain Component']
        _score_data = _score_data[static_columns+[x[0] for x in threshold_columns]].\
                set_index(static_columns).stack().reset_index().\
                rename(columns={'Tissue Category':'region_label',
                                      'Cell Compartment':'feature_label',
                                      'Stain Component':'channel_label',
                                      'level_3':'_temp_ordinal',
                                      0:'threshold_value'})
        _score_data.index.name = 'gate_index'
        _mystats = measurement_statistics
        _score_data['statistic_index'] = _mystats[_mystats['statistic_label']=='Mean'].iloc[0].name
        _thresholds = _score_data.merge(measurement_features.reset_index(),on='feature_label').\
                                  merge(measurement_channels[['channel_label','channel_abbreviation']].reset_index(),on='channel_label').\
                                  merge(regions[['region_label']].reset_index(),on='region_label').\
                                  drop(columns=['feature_label','channel_label','region_label'])
        _thresholds['gate_label'] = _thresholds.apply(lambda x:
                x['channel_abbreviation']+' '+threshold_dict[x['_temp_ordinal']]
            ,1)
        _thresholds = _thresholds.drop(columns=['channel_abbreviation','_temp_ordinal'])
        _thresholds.index.name = 'gate_index'
        return _thresholds

    if is_ordinal:
        raise ValueError("Ordinal processing is not yet tested.")
        _thresholds = _parse_ordinal(_score_data,measurement_statistics,measurement_features,measurement_channels,regions)
    else:
        _thresholds = _parse_binary(_score_data,measurement_statistics,measurement_features,measurement_channels,regions)

        
    ## At this moment we don't support a different threhsold for each region so we will set a nonsense value for the region index... since this threhsold NOT be applied by region
    ##_thresholds.loc[:,'region_index'] = np.nan

    # adding in the drop duplicates to hopefully fix an issue for with multiple tissues
    return _thresholds.drop_duplicates()




