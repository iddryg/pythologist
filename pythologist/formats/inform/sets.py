import os, re, sys, h5py
from pythologist.formats.inform.frame import CellFrameInForm
from pythologist.formats import CellSampleGeneric, CellProjectGeneric
from uuid import uuid4
import pandas as pd


class CellProjectInForm(CellProjectGeneric):
    def __init__(self,h5path,mode='r'):
        super().__init__(h5path,mode)
        return

    def create_cell_sample_class(self):
        return CellSampleInForm()

    def read_path(self,path,project_name=None,name_index=None,verbose=False):
        if project_name is not None: self.set_project_name(project_name)
        if self.mode == 'r': raise ValueError("Error: cannot write to a path in read-only mode.")
        # read all terminal folders as sample_names unless there is none then the sample name is blank
        abspath = os.path.abspath(path)
        if not os.path.isdir(abspath): raise ValueError("Error project path must be a directory")
        sample_dirs = set()
        for root, dirs, files in os.walk(abspath):
            if len(dirs) > 0: continue
            sample_dirs.add(root)
        for s in sample_dirs:
            sname = None
            if name_index is None: sname = s
            else: sname  = os.path.split(s)[name_index]
            sid = self.add_sample_path(s,sample_name=sname,verbose=verbose)
            if verbose: sys.stderr.write("Added sample "+sid+"\n")

    def add_sample_path(self,path,sample_name=None,verbose=False):
        if self.mode == 'r': raise ValueError("Error: cannot write to a path in read-only mode.")
        if verbose: sys.stderr.write("Reading sample "+path+"\n")
        cellsample = self.create_cell_sample_class()
        #print(type(cellsample))
        cellsample.read_path(path,sample_name=sample_name,verbose=verbose)
        cellsample.to_hdf(self.h5path,location='samples/'+cellsample.id,mode='a')
        current = self.key
        if current is None:
            current = pd.DataFrame([{'sample_id':cellsample.id,
                                     'sample_name':cellsample.sample_name}])
            current.index.name = 'db_id'
        else:
            iteration = max(current.index)+1
            addition = pd.DataFrame([{'db_id':iteration,
                                      'sample_id':cellsample.id,
                                      'sample_name':cellsample.sample_name}]).set_index('db_id')
            current = pd.concat([current,addition])
        current.to_hdf(self.h5path,'info',mode='r+',complib='zlib',complevel=9,format='table')
        return cellsample.id

class CellSampleInForm(CellSampleGeneric):
    def __init__(self):
        super().__init__()

    def create_cell_frame_class(self):
        return CellFrameInForm()
    def read_path(self,path,sample_name=None,verbose=False):
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
            tissue_seg_data = tfile if os.path.exists(tfile) else None
            frame = m.group(1).rstrip('_')
            data = os.path.join(path,file)
            if not os.path.exists(summary):
                    if verbose: sys.stderr.write('Missing summary file '+summary+"\n")
                    summary = None
            if not os.path.exists(score):
                    raise ValueError('Missing score file '+score)
            if verbose: sys.stderr.write('Acquiring frame '+data+"\n")
            cid = CellFrameInForm()
            cid.read_raw(frame_name = frame,
                         cell_seg_data_file=data,
                         cell_seg_data_summary_file=summary,
                         score_data_file=score,
                         tissue_seg_data_file=tissue_seg_data,
                         binary_seg_image_file=binary_seg_maps,
                         component_image_file=component_image,
                         verbose=verbose)
            frame_id = cid.id
            self._frames[frame_id]=cid
            frames.append({'frame_id':frame_id,'frame_name':frame,'frame_path':absdir})
        self._key = pd.DataFrame(frames)
        self._key.index.name = 'db_id'
        self.sample_name = sample_name #os.path.split(path)[-1]
