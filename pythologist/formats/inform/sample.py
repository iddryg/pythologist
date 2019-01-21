import os, re, sys
from pythologist.formats.inform.image import CellImageInForm
from pythologist.formats import CellSampleGeneric
from uuid import uuid4
import pandas as pd
class CellSampleInForm(CellSampleGeneric):
    def __init__(self):
        super().__init__()
    @staticmethod
    def create_cell_image_class():
        return CellImageInForm()
    def read_path(self,path,verbose=False):
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
            tissue_seg_data = tfile if os.path.exists(tfile) else None
            frame = m.group(1).rstrip('_')
            data = os.path.join(path,file)
            if not os.path.exists(summary):
                    if verbose: sys.stderr.write('Missing summary file '+summary+"\n")
                    summary = None
            if not os.path.exists(score):
                    raise ValueError('Missing score file '+score)
            if verbose: sys.stderr.write('Acquiring frame '+data+"\n")
            cid = CellImageInForm()
            cid.read_raw(cell_seg_data_file=data,
                         cell_seg_data_summary_file=summary,
                         score_data_file=score,
                         tissue_seg_data_file=tissue_seg_data,
                         binary_seg_image_file=binary_seg_maps,
                         component_image_file=component_image,
                         verbose=verbose)
            frame_id = uuid4().hex
            self._frames[frame_id]=cid
            frames.append({'frame_id':frame_id,'frame_name':frame,'frame_path':absdir})
        self._key = pd.DataFrame(frames)
        self._key.index.name = 'db_id'
