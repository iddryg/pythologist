
from pythologist_reader.formats.inform.sets import CellSampleInForm, CellProjectInForm
from pythologist_reader.formats.inform.custom import CellSampleInFormCustomMask, CellProjectInFormCustomMask
from pythologist_reader.formats.inform.custom import CellSampleInFormLineArea, CellProjectInFormLineArea
import os
from tempfile import NamedTemporaryFile
def read_standard_format_project(path,
                             annotation_strategy,
                             channel_abbreviations={},
                             project_name='myproject',
                             temp_dir=None,
                             custom_mask_name=None,
                             other_mask_name=None,
                             microns_per_pixel=0.496,
                             verbose=False,
                             line_pixel_steps=76
                             ):
    """
    Read a project structed in a standardized format starting from the fold containing a list of samples

    Return a dictinary of CellProjects keyed to the export names
    """
    _check_annotation_strategy(annotation_strategy)
    if temp_dir is not None and not os.path.exists(temp_dir):
    	os.makedirs(temp_dir)

    # Get the export names
    samples = [x for x in os.listdir(path) if x[0]!='.']
    if len(samples) == 0: 
        return None

    for sample in samples:
        sample_folder = os.path.join(path,sample)
        exports = [x for x in os.listdir(os.path.join(sample_folder,'INFORM_ANALYSIS')) if x[0]!='.']
        break

    outputs = {}

    # prepare the exports
    for export in exports:
        temp_path = NamedTemporaryFile(prefix='project-',suffix='.h5',dir=temp_dir,delete=False)
        if annotation_strategy=='GIMP_CUSTOM':
            cpi = CellProjectInFormCustomMask(temp_path.name,mode='w')
            for sample in samples:
                csi = _do_GIMP_CUSTOM(os.path.join(path,sample),export,sample,custom_mask_name,other_mask_name,channel_abbreviations,verbose)
                cpi.append_sample(csi)
        elif annotation_strategy=='GIMP_TSI':
            cpi = CellProjectInFormLineArea(temp_path.name,mode='w')
            for sample in samples:
                csi = _do_GIMP_TSI(os.path.join(path,sample),export,sample,line_pixel_steps,channel_abbreviations,verbose)
                cpi.append_sample(csi)
        else:
            cpi = CellProjectInForm(temp_path.name,mode='w')
            for sample in samples:
                csi = _do_GIMP_OTHER(os.path.join(path,sample),export,sample,channel_abbreviations,verbose)
                cpi.append_sample(csi)
        outputs[export] = cpi
        cpi.project_name = export
        cpi.microns_per_pixel = microns_per_pixel


    return outputs

def _do_GIMP_CUSTOM(base_path,export,sample_name,custom_mask_name,other_mask_name,channel_abbreviations,verbose):
        csi = CellSampleInFormCustomMask()
        csi.read_path(
                      path=os.path.join(base_path,'INFORM_ANALYSIS',export),
                      channel_abbreviations=channel_abbreviations,
                      sample_name=sample_name,
                      verbose = verbose,
                      require = False,
                      require_score = False,
                      custom_mask_name = custom_mask_name,
                      other_mask_name = other_mask_name,
                      alternate_annotation_path = os.path.join(base_path,'ANNOTATIONS')
                     )
        return csi
def _do_GIMP_TSI(base_path,export,sample_name,line_pixel_steps,channel_abbreviations,verbose):
        csi = CellSampleInFormLineArea()
        csi.read_path(
                      path=os.path.join(base_path,'INFORM_ANALYSIS',export),
                      channel_abbreviations=channel_abbreviations,
                      sample_name=sample_name,
                      verbose = verbose,
                      require = False,
                      require_score = False,
                      steps=line_pixel_steps,
                      alternate_annotation_path = os.path.join(base_path,'ANNOTATIONS')
                     )
        return csi
def _do_GIMP_OTHER(base_path,export,sample_name,channel_abbreviations,verbose):
        csi = CellSampleInForm()
        csi.read_path(
                      path=os.path.join(base_path,'INFORM_ANALYSIS',export),
                      channel_abbreviations=channel_abbreviations,
                      sample_name=sample_name,
                      verbose = verbose,
                      require = False,
                      require_score = False,
                      alternate_annotation_path = os.path.join(base_path,'ANNOTATIONS')
                     )
        return csi


def read_standard_format_sample_to_project(path,
                             annotation_strategy,
                             channel_abbreviations={},
	                         sample=None,
                             project_name='myproject',
                             temp_dir=None,
                             custom_mask_name=None,
                             other_mask_name=None,
                             microns_per_pixel=0.496,
                             verbose=False,
                             line_pixel_steps=76
                             ):
    """
    Read a sample structed in a standardized format into a project.  Some advanced CellDataFrame functions may require a project.

    Return a dictinary of CellProjects keyed to the export names
    """
    _check_annotation_strategy(annotation_strategy)
    exports = [x for x in os.listdir(os.path.join(path,'INFORM_ANALYSIS')) if x[0]!='.']
    outputs = {}

    # If we dont supply a sample name assume its the directory being referenced as per standard layout
    if sample is None:
    	_temp, sample = os.path.split(path)

    # prepare the exports
    for export in exports:
        temp_path = NamedTemporaryFile(prefix='project-',suffix='.h5',dir=temp_dir,delete=False)
        if annotation_strategy=='GIMP_CUSTOM':
            cpi = CellProjectInFormCustomMask(temp_path.name,mode='w')
            csi = _do_GIMP_CUSTOM(path,export,sample,custom_mask_name,other_mask_name,channel_abbreviations,verbose)
            cpi.append_sample(csi)
        elif annotation_strategy=='GIMP_TSI':
            cpi = CellProjectInFormLineArea(temp_path.name,mode='w')
            csi = _do_GIMP_TSI(path,export,sample,line_pixel_steps,channel_abbreviations,verbose)
            cpi.append_sample(csi)
        else:
            cpi = CellProjectInForm(temp_path.name,mode='w')
            csi = _do_GIMP_OTHER(path,export,sample,channel_abbreviations,verbose)
            cpi.append_sample(csi)
        cpi.project_name = export
        cpi.microns_per_pixel = microns_per_pixel
        outputs[export] = cpi


    return outputs

def read_standard_format_sample(path,
                             annotation_strategy,
                             channel_abbreviations={},
	                         sample=None,
                             project_name='myproject',
                             custom_mask_name=None,
                             other_mask_name=None,
                             microns_per_pixel=0.496,
                             verbose=False,
                             line_pixel_steps=76
                             ):
    """
    Read a sample structed in a standardized format.

    Return a dictinary of CellSample keyed to the export names
    """
    _check_annotation_strategy(annotation_strategy)
    exports = [x for x in os.listdir(os.path.join(path,'INFORM_ANALYSIS')) if x[0]!='.']
    outputs = {}

    # If we dont supply a sample name assume its the directory being referenced as per standard layout
    if sample is None:
    	_temp, sample = os.path.split(path)

    # prepare the exports
    for export in exports:
        if annotation_strategy=='GIMP_CUSTOM':
            csi = _do_GIMP_CUSTOM(path,export,sample,custom_mask_name,other_mask_name,channel_abbreviations,verbose)
        elif annotation_strategy=='GIMP_TSI':
            csi = _do_GIMP_TSI(path,export,sample,line_pixel_steps,channel_abbreviations,verbose)
        else:
            csi = _do_GIMP_OTHER(path,export,sample,channel_abbreviations,verbose)
        outputs[export] = csi

    return outputs

def _check_annotation_strategy(annotation_strategy):
    if annotation_strategy not in ['GIMP_TSI','GIMP_CUSTOM','INFORM_ANALYSIS','NO_ANNOTATION']:
        raise ValueError("Declare the annotation strategy.")
