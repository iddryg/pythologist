from pythologist.formats.inform.frame import CellFrameInForm
from pythologist.formats.inform.sets import CellSampleInForm, CellProjectInForm
from pythologist.formats.utilities import read_tiff_stack, make_binary_image_array, map_image_ids, watershed_image
from uuid import uuid4
import pandas as pd
import numpy as np
import os, re, sys


class CellProjectInFormLineArea(CellProjectInForm):
    def create_cell_sample_class(self):
        return CellSampleInFormLineArea()


class CellSampleInFormLineArea(CellSampleInForm):
    def create_cell_frame_class(self):
        return CellFrameInFormLineArea()
    def read_path(self,path,sample_name=None,
                            channel_abbreviations=None,
                            verbose=False,require=True,steps=76):
        # Read in a folder of inform cell images
        #
        # These image should be in a format with a image frame name prefix*
        #        *cell_seg_data.txt
        #        *score_data.txt
        #        *cell_seg_data_summary.txt
        #        *tissue_seg_data_summary.txt
        #        *binary_seg_maps.tif
        #        *component.tif
        #
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
            summary = os.path.join(path,m.group(1)+'cell_seg_data_summary.txt')
            binary_seg_maps = os.path.join(path,m.group(1)+'binary_seg_maps.tif')
            component_image = os.path.join(path,m.group(1)+'component_data.tif')
            tfile = os.path.join(path,m.group(1)+'tissue_seg_data.txt')
            tumor = os.path.join(path,m.group(1)+'Tumor.tif')
            margin = os.path.join(path,m.group(1)+'Invasive_Margin.tif')
            tissue_seg_data = tfile if os.path.exists(tfile) else None
            frame = m.group(1).rstrip('_')
            data = os.path.join(path,file)
            if not os.path.exists(summary):
                    if verbose: sys.stderr.write('Missing summary file '+summary+"\n")
                    summary = None
            if not os.path.exists(score):
                    raise ValueError('Missing score file '+score)
            if verbose: sys.stderr.write('Acquiring frame '+data+"\n")
            cid = self.create_cell_frame_class()
            cid.read_raw(frame_name = frame,
                         cell_seg_data_file=data,
                         cell_seg_data_summary_file=summary,
                         score_data_file=score,
                         tissue_seg_data_file=tissue_seg_data,
                         binary_seg_image_file=binary_seg_maps,
                         component_image_file=component_image,
                         channel_abbreviations=channel_abbreviations,
                         verbose=verbose,
                         require=require)
            if verbose: sys.stderr.write("setting tumor and stroma and margin\n")
            cid.set_line_area(margin,tumor,steps=steps,verbose=verbose)
            frame_id = cid.id
            self._frames[frame_id]=cid
            frames.append({'frame_id':frame_id,'frame_name':frame,'frame_path':absdir})
            if verbose: sys.stderr.write("finished tumor and stroma and margin\n")
        self._key = pd.DataFrame(frames)
        self._key.index.name = 'db_id'
        self.sample_name = sample_name #os.path.split(path)[-1]

class CellFrameInFormLineArea(CellFrameInForm):
    def __init__(self):
        super().__init__()
        ### Define extra InForm-specific data tables
        self.data_tables['custom_images'] = {'index':'db_id',
                 'columns':['custom_label','image_id']}
        for x in self.data_tables.keys():
            if x in self._data: continue
            self._data[x] = pd.DataFrame(columns=self.data_tables[x]['columns'])
            self._data[x].index.name = self.data_tables[x]['index']
    def set_line_area(self,line_image,area_image,steps=20,verbose=False):

        #regions = prepare_margin_line_tumor_area(line_image,area_image)
        drawn_binary = read_tiff_stack(line_image)[0]['raw_image']
        drawn_binary = make_binary_image_array(drawn_binary)

        image_id = uuid4().hex
        self._images[image_id] = drawn_binary
        df = pd.DataFrame(pd.Series({'custom_label':'Drawn','image_id':image_id})).T
        df.index.name = 'db_id'
        self.set_data('custom_images',df)

        ids = map_image_ids(drawn_binary,remove_zero=False)
        zeros = ids.loc[ids['id']==0]
        zeros = list(zip(zeros['x'],zeros['y']))
        valid = ids.loc[ids['id']!=0]
        valid = list(zip(valid['x'],valid['y']))
        grown = drawn_binary.copy()
        area_binary = read_tiff_stack(area_image)[0]['raw_image']
        area_binary = make_binary_image_array(area_binary)

        ids = map_image_ids(drawn_binary,remove_zero=False)
        zeros = ids.loc[ids['id']==0]
        zeros = list(zip(zeros['x'],zeros['y']))
        valid = ids.loc[ids['id']!=0]
        valid = list(zip(valid['x'],valid['y']))
        #print(steps)
        grown = watershed_image(drawn_binary,valid,zeros,steps = steps).astype(np.int8)
        processed_image = self.get_image(self.processed_image_id).astype(np.int8)
        #return {'grown':grown,'tumor':area_binary,'processed':processed_image}
        margin_binary = grown&processed_image
        tumor_binary = (area_binary&(~grown))&processed_image
        stroma_binary = (~((tumor_binary|grown)&processed_image))&processed_image

        #regions - we will replace all regions

        d = {'Margin':margin_binary,
                          'Tumor':tumor_binary,
                          'Stroma':stroma_binary}
        self.set_regions(d)
        return d
