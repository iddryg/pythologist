from pythologist_reader.formats.inform.frame import CellFrameInForm, preliminary_threshold_read
from pythologist_reader.formats.inform.sets import CellSampleInForm, CellProjectInForm
from pythologist_reader.formats.inform.custom import CellFrameInFormLineArea, CellFrameInFormCustomMask
import os, re, sys, json
from tempfile import mkdtemp
from glob import glob
from shutil import copytree, copy, rmtree
import pandas as pd
from pythologist_image_utilities import read_tiff_stack, make_binary_image_array, binary_image_dilation
from uuid import uuid4


class CellProjectInFormImmunoProfile(CellProjectInForm):
    """
    Read an ImmunoProfile sample
    """
    def __init__(self,*argv,**kwargs):
        super().__init__(*argv,**kwargs)
        # if we are creating a new project go ahead and give a default name until otherwise set
        if kwargs['mode']=='w': 
            self.project_name = 'ImmunoProfile'
            self.microns_per_pixel = 0.496
        return

    def create_cell_sample_class(self):
        return CellSampleInFormImmunoProfile()

    def add_sample_path(self,path,
                      sample_name=None,
                      verbose=False,
                      invasive_margin_width_microns=40,
                      invasive_margin_drawn_line_width_pixels=10,
                      tumor_stain_name=None,
                      tumor_phenotype_name=None,
                      skip_segmentation_processing=True
                      ):
        """
        Read add a sample in as single project folder and add it to the CellProjectInFormImmunoProfile


        such as ``IP-99-A00001/INFORM_ANALYSIS``:

        | IP-99-A00001/
        | └── INFORM_ANALYSIS
        |     ├── FOXP3
        |     ├── GIMP
        |     └── PD1_PDL1

        Args: 
            path (str): location of the project directory
            sample_name (str): name of the immunoprofile sample (default: rightmost directory in path), can be overridden by 'deidenitfy' set to True .. results in the uuid4 for the sample being used
            export_names (list): specify the names of the exports to read
            channel_abbreviations (dict): dictionary of shortcuts to translate to simpler channel names
            verbose (bool): if true print extra details
            microns_per_pixel (float): conversion factor
            invasive_margin_width_microns (int): size of invasive margin in microns
            invasive_margin_drawn_line_width_pixels (int): size of the line drawn for invasive margins in pixels
            skip_margin (bool): if false (default) read in margin line and define a margin acording to steps.  if true, only read a tumor and stroma.
            skip_segmentation_processing (bool): if false (default) read segementations, else skip to run faster
            deidentify (bool): if false (default) use sample names and frame names derived from the folders.  If true use the uuid4s.

        Returns:
            sample_id, sample_name (tuple) returns the uuid4 assigned as the sample_id, and the sample_name that were given to this sample that was added
        """

        if self.mode == 'r': raise ValueError("Error: cannot write to a path in read-only mode.")
        if sample_name is None: sample_name = os.path.split(path)[-1]


        # fix the margin width
        grow_margin_steps = round(float(invasive_margin_width_microns)/self.microns_per_pixel-float(invasive_margin_drawn_line_width_pixels)/2)
        if verbose: sys.stderr.write("To reach a margin width in each direction of "+str(invasive_margin_width_microns)+"um we will grow the line by "+str(grow_margin_steps)+" pixels\n")


        #if microns_per_pixel is not None: self.microns_per_pixel = microns_per_pixel
        if verbose: sys.stderr.write("microns_per_pixel "+str(self.microns_per_pixel)+"\n")

        # read all terminal folders as sample_names unless there is none then the sample name is blank
        abspath = os.path.abspath(path)
        if not os.path.isdir(abspath): raise ValueError("Error project path must be a directory")
        if os.path.split(abspath)[-1] != 'INFORM_ANALYSIS': raise ValueError("expecting an INFORM_ANALYSIS directory")
        #if len(os.path.split(abspath)) < 2: raise ValueError("expecting an IP path structure")
        #bpath1 = os.path.join(abspath,'INFORM_ANALYSIS')
        #if not os.path.isdir(bpath1): raise ValueError("expecting an INFORM_ANLAYSIS directory as a child directory of IP path")


        #if autodectect_tumor:
        #    # Try to find out what the tumor is on this channel
        #    afiles = os.listdir(os.path.join(bpath1,export_names[0]))
        #    afiles = [x for x in afiles if re.search('_cell_seg_data.txt$',x)]
        #    if len(afiles) == 0: raise ValueError('expected some files in there')
        #    header = list(pd.read_csv(os.path.join(bpath1,export_names[0],afiles[0]),sep="\t").columns)
        #    cell = None
        #    for entry in header:
        #        m = re.match('Entire Cell (.* \('+autodectect_tumor+'\)) Mean \(Normalized Counts, Total Weighting\)',entry)
        #        if m: cell = m.group(1)
        #    if verbose and cell: sys.stderr.write("Detected the tumor channel as '"+str(cell)+"'\n")
        #    if cell: channel_abbreviations[cell] = 'TUMOR'
        #    #print(afile)


        if verbose: sys.stderr.write("Reading sample "+path+" for sample "+sample_name+"\n")

        #errors,warnings = _light_QC(path,export_names,verbose)
        #if len(errors) > 0:
        #    raise ValueError("====== ! Fatal errors encountered in light QC ======\n\n"+"\n".join(errors))

        # Read in one sample FOR this project
        cellsample = self.create_cell_sample_class()
        cellsample.read_path(path,sample_name=sample_name,
                            tumor_stain_name=tumor_stain_name,
                            tumor_phenotype_name=tumor_phenotype_name,
                            verbose=verbose,
                            steps=grow_margin_steps,
                            skip_segmentation_processing=skip_segmentation_processing
                                  )

        # Save the sample TO this project
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
        return cellsample.id, cellsample.sample_name


def _light_QC(path,export_names,verbose):
    errors = []
    warnings = []
    ### Do some light QC to see if we meet assumptions
    if verbose: sys.stderr.write("\n====== Starting light QC check ======\n")

    if verbose: sys.stderr.write("= Check the directory structure \n")
    inform_dir = os.path.join(path,'INFORM_ANALYSIS')
    if not os.path.exists(inform_dir) or not os.path.isdir(inform_dir):
        message = "= ! ERROR: INFORM_ANALYSIS directory not present" + str(export_dir)
        if verbose: sys.stderr.write(message+"\n")
        errors.append(message)
        return (errors,warnings) # can't go on
    else:
        if verbose: sys.stderr.write("=   OK. INFORM_ANALYSIS\n")        
    for export_name in export_names:
        export_dir = os.path.join(path,'INFORM_ANALYSIS',export_name)
        if not os.path.exists(export_dir) or not os.path.isdir(export_dir):
            message = "= ! ERROR: defined export directory not present" + str(export_dir)
            if verbose: sys.stderr.write(message+"\n")
            errors.append(message)
        else:
            if verbose: sys.stderr.write("=   OK. "+str(export_name)+"\n")
    gimp_dir = os.path.join(path,'INFORM_ANALYSIS','GIMP')
    if not os.path.exists(gimp_dir) or not os.path.isdir(gimp_dir):
        message = "= ! ERROR: GIMP directory not present" + str(export_dir)
        if verbose: sys.stderr.write(message+"\n")
        errors.append(message)
    else:
        if verbose: sys.stderr.write("=   OK. GIMP\n")        
    if len(errors) >0: return(errors,warnings)

    if verbose: sys.stderr.write("= Check the cell_seg_data and segmentation from each export\n")
    ### Get file list from the first export
    export_dir = os.path.join(path,'INFORM_ANALYSIS',export_names[0])
    cell_seg_files = set([x for x in os.listdir(export_dir) if re.search('_cell_seg_data.txt$',x)])
    for export_name in export_names[1:]:
        export_dir = os.path.join(path,'INFORM_ANALYSIS',export_name)
        check_files = set([x for x in os.listdir(export_dir) if re.search('_cell_seg_data.txt$',x)])
        if check_files != cell_seg_files:
            message = "= ! ERROR: different cell_seg_data files between "+str(export_names[0])+" and "+str(export_name)+" "+str((cell_seg_files,check_files))
            if verbose: sys.stderr.write(message+"\n")
            errors.append(message)
        else:
            if verbose: sys.stderr.write("=   OK. same cell_seg_data file names between "+str(export_names[0])+" and "+str(export_name)+"\n")
    if len(errors) > 0: return(errors,warnings)
    # Now check the contents
    cell_seg_contents = {}
    def _extract_tuple(mypath):
        data = pd.read_csv(fname,sep="\t")
        return tuple(data[['Cell ID','Cell X Position','Cell Y Position','Phenotype']].sort_values('Cell ID').apply(lambda x: tuple(x),1))
    for cell_seg_file in cell_seg_files:
        fname = os.path.join(path,'INFORM_ANALYSIS',export_names[0],cell_seg_file)
        cell_seg_contents[cell_seg_file] = _extract_tuple(fname)
    for export_name in export_names[1:]:
        for cell_seg_file in cell_seg_files:
            fname = os.path.join(path,'INFORM_ANALYSIS',export_name,cell_seg_file)
            if _extract_tuple(fname) != cell_seg_contents[cell_seg_file]:
                message = "= ! ERROR: different segmentation between by different cell_seg_data "+str((export_names[0],export_name,cell_seg_file))
                if verbose: sys.stderr.write(message+"\n")
                errors.append(message)
            else:
                if verbose: sys.stderr.write("=   OK. Same cell_seg_data between "+str(export_names[0])+" and "+str(export_name)+" for "+str(cell_seg_file)+"\n")
    if len(errors) > 0: return(errors,warnings)

    if verbose: sys.stderr.write("= Check for required files\n")
    base_names = [re.match('(.*)_cell_seg_data.txt$',x).group(1) for x in cell_seg_files]
    for base_name in base_names:
        # For each base name
        failed = False
        # Check tif
        fname = os.path.join(path,'INFORM_ANALYSIS','GIMP',base_name+'_Tumor.tif')
        if not os.path.exists(fname):
            failed = True
            message = "= ! ERROR: missing tumor tif "+str(fname)
            if verbose: sys.stderr.write(message+"\n")
            errors.append(message)
        # Check export contents
        for export_name in export_names:
            fname = os.path.join(path,'INFORM_ANALYSIS',export_name,base_name+'_binary_seg_maps.tif')
            if not os.path.exists(fname):
                failed = True
                message = "= ! ERROR: missing binary_seg_maps tif "+str(fname)
                if verbose: sys.stderr.write(message+"\n")
                errors.append(message)
            fname = os.path.join(path,'INFORM_ANALYSIS',export_name,base_name+'_component_data.tif')
            if not os.path.exists(fname):
                failed = True
                message = "= ! ERROR: missing component_data tif "+str(fname)
                if verbose: sys.stderr.write(message+"\n")
                errors.append(message)
            fname = os.path.join(path,'INFORM_ANALYSIS',export_name,base_name+'_score_data.txt')
            if not os.path.exists(fname):
                failed = True
                message = "= ! ERROR: missing score_data txt "+str(fname)
                if verbose: sys.stderr.write(message+"\n")
                errors.append(message)

        if not failed and verbose: sys.stderr.write("=   OK. required files for "+str(base_name)+"\n")
    if len(errors) > 0: return(errors,warnings)

    if verbose: sys.stderr.write("====== Finished light QC check ======\n\n")
    return (errors,warnings)



class CellSampleInFormImmunoProfile(CellSampleInForm):
    def create_cell_frame_class(self):
        return CellFrameInFormLineArea() # this will be called when we read the HDF
    #def create_cell_frame_class_line_area(self):
    #    return CellFrameInFormLineArea()
    #def create_cell_frame_class_custom_mask(self):
    #    return CellFrameInFormCustomMask()
    def read_path(self,path,sample_name=None,
                            tumor_stain_name=None,
                            tumor_phenotype_name=None,
                            verbose=False,
                            steps=76,
                            skip_segmentation_processing=True
                ):
        if sample_name is None: sample_name = path
        if not os.path.isdir(path):
            raise ValueError('Path input must be a directory')
        if tumor_stain_name is None or tumor_phenotype_name is None:
            raise ValueError("tumor_stain_name and tumor_phenotype_name must be set")
        absdir = os.path.abspath(path)


        frame_prefixes = []
        for _file in glob(os.path.join(path,'PD1_PDL1')+'/*_cell_seg_data.txt'):
            _temp,_prefix = os.path.split(re.match('(.+)_cell_seg_data.txt',_file).group(1))
            frame_prefixes.append(_prefix)

        frames = []
        for _frame_prefix in sorted(frame_prefixes):
            if verbose:
                sys.stderr.write("--- reading frame: "+str(_frame_prefix)+"\n")
            cid = _read_path(path,
                             _frame_prefix,
                             get_strat_dict(tumor_stain_name,tumor_phenotype_name),
                             steps=steps,
                             verbose=verbose,
                             skip_segmentation_processing=skip_segmentation_processing
                             )
            frame_id = cid.id
            self._frames[frame_id]=cid
            frames.append({'frame_id':frame_id,'frame_name':_frame_prefix,'frame_path':path})
            if verbose: sys.stderr.write("finished tumor and stroma and margin\n")
        self._key = pd.DataFrame(frames)
        self._key.index.name = 'db_id'
        self.sample_name = sample_name


def get_strat_dict(tumor_stain_name,tumor_phenotype_name):
    """
    i.e. Cytokeratin (Opal 690), and CYTOKERATIN
    """
    strat = {
    "mutually_exclusive_phenotype_strategies":[
        {
            "phenotype_list":[
                {
                    "assigned_label":"CD8"
                },
                {
                    "assigned_label":tumor_phenotype_name,
                    "label":"TUMOR"
                },
                {
                    "assigned_label":"OTHER"
                }
            ]
        },
    ],
    "channels":[
        {
            "inform_channel_label":"PD-1 (Opal 620)",
            "label":"PD1"
        },
        {
            "inform_channel_label":"PD-L1 (Opal 520)",
            "label":"PDL1"
        },
        {
            "inform_channel_label": "DAPI",
            "label":"DAPI"
        },
        {
            "inform_channel_label": "CD8 (Opal 480)",
            "label":"CD8"
        },
        {
            "inform_channel_label": "Foxp3 (Opal 570)",
            "label":"FOXP3"
        },
        {
            "inform_channel_label": tumor_stain_name,
            "label":"TUMOR"
        },
        {
            "inform_channel_label": "CD68 (Opal 780)",
            "label":"CD68"
        }
    ]
    }
    def _deepcopy(d):
        return json.loads(json.dumps(d))
    strat_dict = {}
    strat_dict['PD1_PDL1'] = _deepcopy(strat)
    strat_dict['FOXP3'] = _deepcopy(strat)

    for x in strat_dict['PD1_PDL1']['channels']:
        if x['label'] in ["PD1","PDL1"]:
            x['analyze_threshold'] = True

    for x in strat_dict['FOXP3']['channels']:
        if x['label'] in ["FOXP3"]:
            x['analyze_threshold'] = True
    return strat_dict

def _read_export(path,frame_name,export_name,strat_dict,steps=76,verbose=False,skip_segmentation_processing=False):
    if verbose: sys.stderr.write("Processing export: "+str(export_name)+"\n")
    cfi = CellFrameInFormLineArea()
    _export_prefix = os.path.join(path,export_name,frame_name)
    cfi.read_raw(
        frame_name = frame_name,
        cell_seg_data_file = _export_prefix+'_cell_seg_data.txt',
        score_data_file = _export_prefix+'_score_data.txt',
        binary_seg_image_file = _export_prefix+'_binary_seg_maps.tif',
        #component_image_file = base_path+'component_data.tif',
        inform_analysis_dict = strat_dict[export_name],
        verbose = verbose,
        require_component = False,
        require_score = True,
        dry_run = False,
        skip_segmentation_processing=skip_segmentation_processing
    )
    cfi.set_line_area(
        os.path.join(path,'GIMP',frame_name+'_Invasive_Margin.tif'),
        os.path.join(path,'GIMP',frame_name+'_Tumor.tif'),
        steps = steps,
        verbose = verbose
    )
    cfi.microns_per_pixel = 0.496
    return cfi

def _read_path(path,frame_name,strat_dict,steps=76,verbose=False,skip_segmentation_processing=False):
    _mutually_exclusive_phenotypes = ['CD8','TUMOR','OTHER']
    _e1 = _read_export(path,frame_name,'PD1_PDL1',strat_dict,steps=steps,verbose=verbose,skip_segmentation_processing=skip_segmentation_processing)
    _e2 = _read_export(path,frame_name,'FOXP3',strat_dict,steps=steps,verbose=verbose,skip_segmentation_processing=skip_segmentation_processing)

    # get the feature tables from FOXP3
    _features = _e2.get_data('features')
    _features = _features.loc[_features['feature_label']=='FOXP3',:]
    _features_index = _features.iloc[0].name
    _feature_definitions = _e2.get_data('feature_definitions')
    _feature_definitions = _feature_definitions.loc[_feature_definitions['feature_index']==_features_index,:] 
    _feature_definitions_indecies = [x for x in _feature_definitions.index]
    _cell_features = _e2.get_data('cell_features')
    _cell_features = _cell_features.loc[_cell_features['feature_definition_index'].isin(_feature_definitions_indecies),:]
    _t2 = _features.merge(_feature_definitions,left_index=True,right_on='feature_index').\
       merge(_cell_features,left_index=True,right_on='feature_definition_index',)

    # get the maximum feature indecies from PD1_PDL1
    features_iter = _e1.get_data('features').index.max()+1
    feature_definitions_iter = _e1.get_data('feature_definitions').index.max()+1
    cell_features_iter = _e1.get_data('cell_features').index.max()+1

    # make a big table
    _t3 = _t2.copy().reset_index()
    _t3['feature_index'] = _t3['feature_index'].apply(lambda x: x+features_iter)
    _t3['feature_definition_index'] = _t3['feature_definition_index'].apply(lambda x: x+feature_definitions_iter)
    _t3['db_id'] = _t3['db_id'].apply(lambda x: x+cell_features_iter)

    # shift indecies so we can merge features
    _cf = _t3[_cell_features.reset_index().columns].set_index('db_id')
    _fd = _t3[_feature_definitions.reset_index().columns].drop_duplicates().\
        set_index('feature_definition_index')
    _fs = _t3[_features.reset_index().columns].drop_duplicates().\
        set_index('feature_index')

    # merge our tables
    _e1.set_data('cell_features',pd.concat([_e1.get_data('cell_features'),_cf]))
    _e1.set_data('feature_definitions',pd.concat([_e1.get_data('feature_definitions'),_fd]))
    _e1.set_data('features',pd.concat([_e1.get_data('features'),_fs]))
    return _e1


    #_cdf1 = _e1.cdf(region_group='InFormLineArea',mutually_exclusive_phenotypes=_mutually_exclusive_phenotypes)
    #_cdf2 = _e2.cdf(region_group='InFormLineArea',mutually_exclusive_phenotypes=_mutually_exclusive_phenotypes)
    #_cdf, _f = _cdf1.merge_scores(_cdf2)
    #if _f.shape[0] > 0:
    #    raise ValueError("Mismatched segmentation of size: "+str(_f.shape[0]))
    #return _cdf