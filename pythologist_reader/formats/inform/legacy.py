from pythologist_reader.formats.inform.frame import CellFrameInForm
from pythologist_reader.formats.inform.sets import CellSampleInForm, CellProjectInForm
from pythologist_image_utilities import read_tiff_stack, make_binary_image_array
import os, re, sys
from uuid import uuid4
import pandas as pd
import numpy as np

class CellProjectInFormSeperateSegmentations(CellProjectInForm):
    """
    Read in a project that has nuclear and membrane segmentations in seperate files

    Accessed via ``read_path`` with the additonal parameters
    """
    def read_path(self,*args,**kwargs):
        """
        Read in the project folder

        Args: 
            path (str): location of the project directory
            project_name (str): name of the project
            sample_name_index (int): where in the directory chain is the foldername that is the sample name if not set use full path.  -1 is last directory
            channel_abbreviations (dict): dictionary of shortcuts to translate to simpler channel names
            verbose (bool): if true print extra details
            require (bool): if true (default), require that channel componenet image be present
            microns_per_pixel (float): conversion factor
        """
        super().read_path(*args,**kwargs)

    def create_cell_sample_class(self):
        return CellSampleInFormSeperateSegmentations()

class CellSampleInFormSeperateSegmentations(CellSampleInForm):
    def create_cell_frame_class(self):
        return CellFrameInFormSeperateSegmentations()
    def read_path(self,path,sample_name=None,
                            channel_abbreviations=None,
                            verbose=False,require=True,require_score=True):
        if sample_name is None: sample_name = path
        if not os.path.isdir(path):
            raise ValueError('Path input must be a directory')
        absdir = os.path.abspath(path)
        z = 0
        files = os.listdir(path)
        z += 1
        segs = [x for x in files if re.search('_cell_seg_data.txt$',x)]
        if len(segs) == 0: raise ValueError("There needs to be cell_seg_data in the folder.")
        frames = []
        for file in segs:
            m = re.match('(.*)cell_seg_data.txt$',file)
            score = os.path.join(path,m.group(1)+'score_data.txt')
            memb_seg_map = os.path.join(path,m.group(1)+'memb_seg_map.tif')
            nuc_seg_map = os.path.join(path,m.group(1)+'nuc_seg_map.tif')
            #binary_seg_maps = os.path.join(path,m.group(1)+'binary_seg_maps.tif')
            component_image = os.path.join(path,m.group(1)+'component_data.tif')
            tfile = os.path.join(path,m.group(1)+'tissue_seg_data.txt')
            tissue_seg_data = tfile if os.path.exists(tfile) else None
            frame = m.group(1).rstrip('_')
            data = os.path.join(path,file)
            if not os.path.exists(score):
                    raise ValueError('Missing score file '+score)
            if verbose: sys.stderr.write('Acquiring frame '+data+"\n")
            cid = self.create_cell_frame_class()
            cid.read_raw(frame_name = frame,
                         cell_seg_data_file=data,
                         score_data_file=score,
                         tissue_seg_data_file=tissue_seg_data,
                         memb_seg_image_file=memb_seg_map,
                         nuc_seg_image_file=nuc_seg_map,
                         component_image_file=component_image,
                         channel_abbreviations=channel_abbreviations,
                         verbose=verbose,
                         require=require,require_score=require_score)
            if verbose: sys.stderr.write("setting mask and not mask\n")
            frame_id = cid.id
            self._frames[frame_id]=cid
            frames.append({'frame_id':frame_id,'frame_name':frame,'frame_path':absdir})
            if verbose: sys.stderr.write("finished mask and not mask\n")
        self._key = pd.DataFrame(frames)
        self._key.index.name = 'db_id'
        self.sample_name = sample_name


class CellFrameInFormSeperateSegmentations(CellFrameInForm):
    def __init__(self):
        super().__init__()
    def read_raw(self,
                 frame_name = None,
                 cell_seg_data_file=None,
                 score_data_file=None,
                 tissue_seg_data_file=None,
                 memb_seg_image_file=None,
                 nuc_seg_image_file=None,
                 component_image_file=None,
                 verbose=False,
                 channel_abbreviations=None,
                 require=True,require_score=True):
        self.frame_name = frame_name
        ### Read in the data for our object
        if verbose: sys.stderr.write("Reading text data.\n")
        self._read_data(cell_seg_data_file,
                   score_data_file,
                   tissue_seg_data_file,
                   verbose,
                   channel_abbreviations,require=require,require_score=require_score)
        if verbose: sys.stderr.write("Reading image data.\n")
        self._read_images(memb_seg_image_file,nuc_seg_image_file,
                   component_image_file,
                   verbose=verbose,
                   require=require)
        return

    ### Lets work with image files now
    def _read_images(self,memb_seg_image_file=None,nuc_seg_image_file=None,
                     component_image_file=None,verbose=False,require=True):
        # Start with the binary seg image file because if it has a processed image area,
        # that will be applied to all other masks and we can get that segmentation right away

        # Now we've read in whatever we've got fromt he binary seg image
        if verbose: sys.stderr.write("Reading component images.\n")
        if require or (not require and component_image_file and os.path.isfile(component_image_file)): 
            self._read_component_image(component_image_file)
        if verbose: sys.stderr.write("Finished reading component images.\n")

        if memb_seg_image_file is None:
            raise ValueError("missing membrane seg")
        if nuc_seg_image_file is None: 
            raise ValueError("missing nuclear seg")


        if verbose: sys.stderr.write("Binary seg file present.\n")
        self._read_seg_images(memb_seg_image_file,nuc_seg_image_file)
        segmentation_images = self.get_data('segmentation_images').set_index('segmentation_label')
        if 'Nucleus' in segmentation_images.index and \
               'Membrane' in segmentation_images.index:
                if verbose: sys.stderr.write("Making cell-map filled-in.\n")
                ## See if we are a legacy membrane map
                mem = self._images[self.get_data('segmentation_images').\
                          set_index('segmentation_label').loc['Membrane','image_id']]
                if verbose: sys.stderr.write("making legacy cell map\n")
                self._make_cell_map_legacy()
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

    def _read_seg_images(self,memb_seg_image_file,nuc_seg_image_file):
        segmentation_names = []
        memb = make_binary_image_array(read_tiff_stack(memb_seg_image_file)[0]['raw_image'])
        memb_id = uuid4().hex
        self._images[memb_id] = memb.astype(int)
        segmentation_names.append(['Membrane',memb_id])
        nuc = read_tiff_stack(nuc_seg_image_file)[0]['raw_image']
        nuc_id = uuid4().hex
        self._images[nuc_id] = nuc.astype(int)
        segmentation_names.append(['Nucleus',nuc_id])
        #_mask_key = pd.DataFrame(mask_names,columns=['mask_label','image_id'])
        #_mask_key.index.name = 'db_id'
        #self.set_data('mask_images',_mask_key)
        _segmentation_key = pd.DataFrame(segmentation_names,columns=['segmentation_label','image_id'])
        _segmentation_key.index.name = 'db_id'
        self.set_data('segmentation_images',_segmentation_key)

    def _read_component_image(self,filename):
        stack = read_tiff_stack(filename)
        channels = []
        for i,raw in enumerate(stack):
            meta = raw['raw_meta']['image_description']
            markers = [x.split('=')[1] for x in meta.split('\n')[1:]]
            channel_label = markers[i]
            image_id = uuid4().hex
            self._images[image_id] = raw['raw_image'].astype(self._storage_type)
            channels.append((channel_label,image_id))
        df = pd.DataFrame(channels,columns=['channel_label','image_id'])
        temp = self.get_data('measurement_channels').drop(columns=['image_id']).reset_index().merge(df,on='channel_label',how='left')
        self.set_data('measurement_channels',temp.set_index('channel_index'))
        return
