import os, re, sys, h5py
from pythologist_reader.formats.inform.frame import CellFrameInForm
from pythologist_reader import CellSampleGeneric, CellProjectGeneric
from pythologist_reader.qc import QCTestGeneric, QC, Result
from uuid import uuid4
import pandas as pd


class InFormQC(QC):
    """
    Extend the QC object to add format specific tests
    """
    def __init__(self,proj,*args,**kwargs):
        super().__init__(proj,*args,**kwargs)
        self._test_list = self._test_list+\
            [QCCompartmentConsistency

            ]
    def channel_histograms(self,minvalue=None,maxvalue=None,bins=100):
        if self.verbose: sys.stderr.write("getting parent histograms\n")
        hist = super().channel_histograms(minvalue=minvalue,maxvalue=maxvalue,bins=bins)
        gates = self.proj.gates
        cnames = ['project_id','project_name','sample_id','sample_name','frame_id','frame_name','channel_label','channel_abbreviation']
        hist2 = hist.merge(gates,on=cnames,how='left')
        #only show one threshold per image/channel
        df = hist2.groupby(cnames+['bins']).first().reset_index().drop(columns=['gate_label','region_label'])
        return df

class QCCompartmentConsistency(QCTestGeneric):
    @property
    def name(self): return 'Check if stains are scored across the same compartment'
    def run(self):
        gates = self.proj.gates
        check = gates[['sample_name','sample_id','frame_name','frame_id','region_label','gate_label','feature_label']].drop_duplicates()
        counts = check[['gate_label','feature_label']].drop_duplicates()
        counts = counts.groupby('gate_label').apply(lambda x: list(x['feature_label'])).reset_index().rename(columns={0:'feature_labels'})
        counts['count'] = counts.apply(lambda x: len(x['feature_labels']),1)
        if counts.loc[counts['count']>1].shape[0] > 0:
            return Result(result='WARNING',
                      about='the thresholding of the following phentypes took place in different cellular compartments depending on the sample/image/region'+\
                            str(counts.loc[counts['count']>1]),
                      count = None,
                      total=None)
        return Result(result='PASS',
                      about='thresholding of each scored name is done consistently in the same compartments',
                      count = None,
                      total=None)



class CellProjectInForm(CellProjectGeneric):
    def __init__(self,h5path,mode='r'):
        super().__init__(h5path,mode)
        return

    def create_cell_sample_class(self):
        return CellSampleInForm()

    def read_path(self,path,
                       project_name=None,
                       sample_name_index=None,
                       channel_abbreviations=None,
                       verbose=False,
                       require=True,
                       require_score=True,
                       skip_segmentation_processing=False,
                       microns_per_pixel=None,
                       **kwargs):
        """
        Read in the project folder

        Args: 
            path (str): location of the project directory
            project_name (str): name of the project
            sample_name_index (int): where in the directory chain is the foldername that is the sample name if not set use full path.  -1 is last directory
            channel_abbreviations (dict): dictionary of shortcuts to translate to simpler channel names
            verbose (bool): if true print extra details
            require (bool): if true (default), require that channel componenet image be present
            require_score (bool): if true (default), require there be a score file in the data
            skip_segmentation_processing (bool): if false (default), it will store the cellmap and edgemap images, if true, it will skip these steps to save time but downstream applications will not be able to generate the cell-cell contact measurements or segmentation images.
            microns_per_pixel (float): conversion factor
        """
        if project_name is not None: self.project_name = project_name
        if microns_per_pixel is not None: self.microns_per_pixel = microns_per_pixel
        if verbose: sys.stderr.write("microns_per_pixel "+str(self.microns_per_pixel)+"\n")
        if self.mode == 'r': raise ValueError("Error: cannot write to a path in read-only mode.")
        # read all terminal folders as sample_names unless there is none then the sample name is blank
        abspath = os.path.abspath(path)
        if not os.path.isdir(abspath): raise ValueError("Error project path must be a directory")
        sample_dirs = set()
        for root, dirs, files in os.walk(abspath):
            files = [f for f in files if not f[0] == '.']
            dirs[:] = [d for d in dirs if not d[0] == '.']
            if len(dirs) > 0: continue
            sample_dirs.add(root)
        for s in sample_dirs:
            sname = None
            if sample_name_index is None: sname = s
            else: sname  = s.split(os.sep)[sample_name_index]
            sid = self.add_sample_path(s,sample_name=sname,
                                         channel_abbreviations=channel_abbreviations,
                                         verbose=verbose,require=require,
                                         require_score=require_score,
                                         skip_segmentation_processing=skip_segmentation_processing,
                                         **kwargs)
            if verbose: sys.stderr.write("Added sample "+sid+"\n")


    def add_sample_path(self,path,sample_name=None,channel_abbreviations=None,
                                  verbose=False,require=True,require_score=True,skip_segmentation_processing=False,**kwargs):
        if self.mode == 'r': raise ValueError("Error: cannot write to a path in read-only mode.")
        if verbose: sys.stderr.write("Reading sample "+path+" for sample "+sample_name+"\n")
        cellsample = self.create_cell_sample_class()
        #print(type(cellsample))
        cellsample.read_path(path,sample_name=sample_name,
                                  channel_abbreviations=channel_abbreviations,
                                  verbose=verbose,require=require,
                                  require_score=require_score,
                                  skip_segmentation_processing=skip_segmentation_processing,
                                  **kwargs)
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

    ## Put some custom accessors here specific to InForm projects

    def qc(self,*args,**kwargs):
        return InFormQC(self,*args,**kwargs)

    @property
    def gates(self):
        """
        Get all the gates from the frames / samples in the project
        """
        def _get_gates(f):
            thresh = f.get_data('thresholds')
            mf = f.get_data('measurement_features')
            mc = f.get_data('measurement_channels')
            mr = f.get_data('regions')
            thresh = thresh.merge(mf,left_on='feature_index',right_index=True).\
                merge(mc,left_on='channel_index',right_index=True).\
                merge(mr[['region_label']],left_on='region_index',right_index=True)
            thresh = thresh.loc[:,~thresh.columns.str.contains('index')].drop(columns=['image_id'])
            return thresh
        pname = self.project_name
        pid = self.id
        allgates = []
        for s in self.sample_iter():
            sname = s.sample_name
            sid = s.id
            for f in s.frame_iter():
                fname = f.frame_name
                fid = f.id
                gates = _get_gates(f)
                gates['project_name'] = pname
                gates['project_id'] = pid
                gates['sample_name'] = sname
                gates['sample_id'] = sid
                gates['frame_name'] = fname
                gates['frame_id'] = fid
                allgates.append(gates)
        allgates = pd.concat(allgates).reset_index(drop=True)
        return allgates

class CellSampleInForm(CellSampleGeneric):
    def __init__(self):
        super().__init__()

    def create_cell_frame_class(self):
        return CellFrameInForm()

    def read_path(self,path,sample_name=None,
        channel_abbreviations=None,verbose=False,require=True,require_score=True,skip_segmentation_processing=False,**kwargs):
        """
        Read in the project folder

        Args: 
            path (str): location of the project directory
            project_name (str): name of the project
            sample_name_index (int): where in the directory chain is the foldername that is the sample name if not set use full path.  -1 is last directory
            channel_abbreviations (dict): dictionary of shortcuts to translate to simpler channel names
            verbose (bool): if true print extra details
            require (bool): if true (default), require that channel componenet image be present
            require_score (bool): if true (default), require that score file be present
            skip_segmentation_processing (bool): if false (default), it will store the cellmap and edgemap images, if true, it will skip these steps to save time but downstream applications will not be able to generate the cell-cell contact measurements or segmentation images.
            microns_per_pixel (float): conversion factor
        """
        if sample_name is None: sample_name = path
        if not os.path.isdir(path):
            raise ValueError('Path input must be a directory')
        absdir = os.path.abspath(path)
        z = 0
        files = [x for x in os.listdir(path) if not x[0]=='.']
        z += 1
        segs = [x for x in files if re.search('_cell_seg_data.txt$',x)]
        if len(segs) == 0: raise ValueError("There needs to be cell_seg_data in the folder.")
        frames = []
        for file in segs:
            m = re.match('(.*)cell_seg_data.txt$',file)
            score = os.path.join(path,m.group(1)+'score_data.txt')
            #summary = os.path.join(path,m.group(1)+'cell_seg_data_summary.txt')
            binary_seg_maps = os.path.join(path,m.group(1)+'binary_seg_maps.tif')
            component_image = os.path.join(path,m.group(1)+'component_data.tif')
            tfile = os.path.join(path,m.group(1)+'tissue_seg_data.txt')
            tissue_seg_data = tfile if os.path.exists(tfile) else None
            frame = m.group(1).rstrip('_')
            data = os.path.join(path,file)

            if not os.path.exists(score) and require_score:
                raise ValueError('Missing score file '+score)
            elif not os.path.exists(score):
                score = None
            if verbose: sys.stderr.write('Acquiring frame '+data+"\n")
            cid = self.create_cell_frame_class()
            cid.read_raw(frame_name = frame,
                         cell_seg_data_file=data,
                         score_data_file=score,
                         tissue_seg_data_file=tissue_seg_data,
                         binary_seg_image_file=binary_seg_maps,
                         component_image_file=component_image,
                         channel_abbreviations=channel_abbreviations,
                         verbose=verbose,
                         require=require,
                         require_score=require_score,
                         skip_segmentation_processing=skip_segmentation_processing)
            frame_id = cid.id
            self._frames[frame_id]=cid
            frames.append({'frame_id':frame_id,'frame_name':frame,'frame_path':absdir})
        self._key = pd.DataFrame(frames)
        self._key.index.name = 'db_id'
        self.sample_name = sample_name #os.path.split(path)[-1]
