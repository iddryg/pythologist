"""
Read InForm files into the appropriate Schema and return the json object

        Args:
            phenotypes (list): a list of phenotypes to add to scored calls.  if none or not set, add them all
            overwrite (bool): if True allow the overwrite of a phenotype, if False, the phenotype must not exist in the scored calls
        Returns:
            CellDataFrame


InForm files 

a. Make sure the project directy exists
b. Make sure only there are only folders among the visible files
c. Make sure the visible folders are only exactly named as those among the sample manifest
d. Make sure within each sample folder that all expected Exports exist
e. Make sure there aren't more folders burried deeper in these .. only files
f. Make sure all files present start with a prefix of the sample whose folder they are in that is followed by an underscore
g. Make sure there are not image frames that do not have a cell seg data since this is how we find what ROIs we have
h. Make sure that the required images for each annotation strategy are present

"""
import os, re, time, stat, hashlib, logging
from importlib_resources import files
from pythologist.schemas import get_validator

# Lets preload our validators for relevent schemas as globals
project_schema_validator = get_validator(files('schema_data.inputs.platforms.InForm').joinpath('project.json'))
analysis_schema_validator = get_validator(files('schema_data.inputs.platforms.InForm').joinpath('analysis.json'))
files_schema_validator = get_validator(files('schema_data.inputs.platforms.InForm').joinpath('files.json'))


def injest_project(project_json,analysis_json,project_directory):
    """
    Read a path pointing to multiple InForm sample folders

    Args:
        project_json (dict): The json object as a valid project schema
        analysis_json (dict): The json object as a valid analysis schema
    Returns:
        samples (list): A list of json objects for the samples in the project
    """

    # Might want to revalidate teh project_schema here against the project schema
    project_schema_validator.validate(project_json)
    analysis_schema_validator.validate(analysis_json)

    # a. Make sure the project directory exists
    if not os.path.exists(project_directory):
        raise ValueError('Project directory "'+str(project_directory)+'" does not exist.')
    if not os.path.isdir(project_directory):
        raise ValueError('Project directory path "'+str(project_directory)+'" does not a directory.')

    # c. Look for the SAMPLES folder in the project directory
    project_samples_directory = os.path.join(project_directory,'SAMPLES')
    if not os.path.exists(project_samples_directory) and not os.path.isdir(project_samples_directory):
        raise ValueError('Project samples path '+str(project_directory)+' does not have a SAMPLES directory')

    # b. Make sure the project directories visible files are only folders

    folder_names = [x for x in os.listdir(project_samples_directory) if x[0]!='.'] 
    for folder_name in folder_names:
        if not os.path.isdir(os.path.join(project_samples_directory,folder_name)):
            raise ValueError('File present in the project directory that is not a folder '+str(folder_name))

    # c. Make sure the visible folders are only exactly named as those among the sample manifest

    sample_allow_list = [x['sample'] for x in project_json['samples']]

    unwelcome = sorted(list(set(folder_names) - set(sample_allow_list)))

    if len(unwelcome) > 0:
        raise ValueError('Folder(s) are present in the project folder that are not listed in the sample manifest '+str(unwelcome))

    samples = []
    injest_success = True
    injest_errors = []
    for sample_path, sample_name in [(os.path.join(project_directory,x),x) for x in folder_names]:
        #print(sample_name)
        sample_json, injest_success0, injest_errors0 = injest_sample(sample_name,project_json,analysis_json,project_directory)
        injest_errors += injest_errors0
        injest_success  = injest_success0 and injest_success
        samples.append(sample_json)
    return samples, injest_success, injest_errors



def injest_sample(sample_name,project_json,analysis_json,project_directory):
    """
    Read a path pointing to an InForm sample folder

    Args:
        phenotypes (list): a list of phenotypes to add to scored calls.  if none or not set, add them all
        overwrite (bool): if True allow the overwrite of a phenotype, if False, the phenotype must not exist in the scored calls
    Returns:
        CellDataFrame
    """


    
    # Confirm the inputs are valid
    project_schema_validator.validate(project_json)
    analysis_schema_validator.validate(analysis_json)
    # a. Make sure the project directory exists
    if not os.path.exists(project_directory):
        raise ValueError('Project directory "'+str(project_directory)+'" does not exist.')
    if not os.path.isdir(project_directory):
        raise ValueError('Project directory path "'+str(project_directory)+'" does not a directory.')
    sample_path = os.path.join(project_directory,'SAMPLES',sample_name)
    if not os.path.exists(sample_path):
        raise ValueError(str(sample_path)+' sample path does not exist')
    if sample_name not in [x['sample'] for x in project_json['samples']]:
        raise ValueError(str(sample_name)+ ' not in sample allow list from project definition')



    # d. Make sure all export folders and only those export folders are present in each sample folder

    expected_export_files = [x['export_name'] for x in analysis_json['inform_exports']]

    observed_files = [x for x in os.listdir(os.path.join(sample_path,'INFORM_ANALYSIS')) if x[0]!='.']
    for observed_file in observed_files:
        if not os.path.isdir(os.path.join(sample_path,'INFORM_ANALYSIS',observed_file)):
            raise ValueError('File is present in the Sample folder that is not an InForm Export Folder '+str(observed_file)+' is not among '+str(expected_export_files))
    missing = sorted(list(set(expected_export_files)-set(observed_files)))
    if len(missing) > 0:
        raise ValueError('InForm Export Folder(s) are missing from the sample path "'+sample_path+'" that are expected '+str(missing))
    unwelcome = sorted(list(set(observed_files)-set(expected_export_files)))
    if len(unwelcome) > 0:
        raise ValueError('Unexpected folder(s) are present in the sample path "'+sample_path+'" '+str(unwelcome))


    # Set up the output file
    sample_files = {
            'sample_name':sample_name,
            'sample_directory':sample_path,
            'exports':[
            ]
    }
    for export_path, export_name in [(os.path.join(sample_path,'INFORM_ANALYSIS',x),x) for x in expected_export_files]:
        # Inspect the export_path to make sure there isnt something funky inside of it
    
        # e. Make sure there aren't more folders burried deeper in these .. only files

        # f. Make sure all files present start with a prefix of the sample whose folder they are in that is followed by an underscore
        _inspect_export_folder(export_path,sample_name,analysis_json)

        export = {
            'export_name':export_name,
            'images':_do_export_images(export_path,sample_name,analysis_json,sample_path)
        }
        sample_files['exports'].append(export)

    files_schema_validator.validate(instance=sample_files)
    return sample_files, True, []

def _inspect_export_folder(export_path,sample_name,analysis_json):
    logger = logging.getLogger("insepct export folder")
    prog = re.compile('^(.+)_cell_seg_data\.txt$')
    observed_files = [x for x in os.listdir(export_path) if x[0]!='.']
    # Now look for individual frames
    frame_names = [prog.match(x).group(1) for x in observed_files if prog.match(x)]


    # figure out if we are seeing
    unwanted_folders = [x for x in observed_files if os.path.isdir(os.path.join(export_path,x))]
    if len(unwanted_folders) > 0:
        raise ValueError('Unexpected folder(s) present in the export folder '+str(unwanted_folders))

    # See if there are any files present that are not part of an image with a cell seg data
    for file_name in observed_files:
        if len([x for x in frame_names if file_name.startswith(x+"_")])==0:
            raise ValueError("Found an unexpected file "+str(file_name)+" that isnt prefixed the same as the cell_seg_data files.")


    _temp, export_name = os.path.split(export_path)
    misnamed = [x for x in frame_names if not x.startswith(sample_name+'_')]
    if len(misnamed) > 0:
        logger.warning('Image file(s) do not start with the anticipated sample name "'+str(export_name)+'" "'+str(sample_name)+'" '+str(misnamed))

    # g. Make sure there are not image frames that do not have a cell seg data
    all_files = set(observed_files.copy())
    for file_name in list(all_files):
        for frame_name in frame_names:
            if file_name.startswith(frame_name):
                all_files-=set([file_name])
    if len(all_files) > 0:
        raise ValueError('It appears there may be some image ROIs that are missing their cell_seg_data.txt file '+str(all_files))

def _sha256(fname):
    hash_sha256 = hashlib.sha256()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_sha256.update(chunk)
    return hash_sha256.hexdigest()

# Now for each sample read it into the files json-schema object
def _do_export_images(export_path,sample_name,analysis_json,sample_path):


    # Some things we need to check 
    # get the image_file list
    image_files = [x for x in os.listdir(export_path) if x[0]!='.']    

    # This will bey keyed by each of the image names and contain a dictionary of the primary images we are putting together pointing to their path
    image_obj = {}

    # Get the various core image types we are interested in collecting from our image data from the schema itself.
    for image_file_name in files_schema_validator.schema['definitions']['image_data']['properties']:
        image_file = files_schema_validator.schema['definitions']['image_data']['properties'][image_file_name]
        suffix = image_file['allOf'][1]['properties']['suffix']['const']
        rgxstr = suffix.replace('.','\.')
        prog = re.compile('^(.+)_'+rgxstr+'$')
        _dict = dict([(prog.match(x).group(1),x) for x in image_files if prog.match(x)])
        for image_name in _dict.keys():
            if image_name not in image_obj:
                image_obj[image_name] = {}
            image_obj[image_name][image_file_name] = _dict[image_name]
    image_output = {}
    for image_name in sorted(list(image_obj.keys())):
        #print(image_name)
        _img = {
            'image_name':image_name,
            'image_data':{},
            'image_annotations':[]
        }
        for image_file_name in image_obj[image_name]:
            file_path = os.path.join(export_path,image_obj[image_name][image_file_name])
            _img['image_data'][image_file_name] = _generate_file_dictionary(file_path)
        #print(_img)
        image_output[image_name] = _img

    # Load annotations if needed    
    # check against annotation_strategy_requirements
    annotation_strategy = analysis_json['parameters']['region_annotation_strategy']
    for image_name in sorted(list(image_obj.keys())):
        if annotation_strategy == 'GIMP_TSI':
            #print(image_output[image_name])
            image_output[image_name]['image_annotations'] = _do_region_annotation_GIMP_TSI(image_name,sample_path,image_files)
        elif annotation_strategy == 'GIMP_CUSTOM':
            image_output[image_name]['image_annotations'] = _do_region_annotation_GIMP_CUSTOM(image_name,
            	                                                                              sample_path,image_files,
            	                                                                              analysis_json['parameters']['region_annotation_custom_label']
            	                                                                              )
        elif annotation_strategy == 'INFORM_ANALYSIS':
            continue
        elif annotation_strategy == 'NO_ANNOTATION':
            continue
        else:
            raise ValueError("Unsupported Region Annotation Strategy. You shouldn't see this because of enum check")

    return list(image_output.values())

def _do_region_annotation_GIMP_TSI(image_name,sample_path,image_files):
    annotation_folder = os.path.join(sample_path,'ANNOTATIONS')
    if not os.path.exists(annotation_folder) or not os.path.isdir(annotation_folder):
        raise ValueError("Unable to locate ANNOTATIONS folder in the sample folder "+str(sample_path))
    prospective1 = os.path.join(annotation_folder,image_name+'_Tumor.tif')
    outputs = []
    if not os.path.exists(prospective1):
        raise ValueError("Required file is not present. "+str(prospective1))
    d1 =  _generate_file_dictionary(prospective1)
    d1['mask_label'] = 'Tumor'
    outputs += [d1]

    # This file is not required
    prospective2 = os.path.join(annotation_folder,image_name+'_Invasive_Margin.tif')
    if os.path.exists(prospective2):
        d2 =  _generate_file_dictionary(prospective2)
        d2['mask_label'] = 'TSI Line'
        outputs += [d2]
    return outputs

def _do_region_annotation_GIMP_CUSTOM(image_name,sample_path,image_files,custom_label):
    annotation_folder = os.path.join(sample_path,'ANNOTATIONS')
    if not os.path.exists(annotation_folder) or not os.path.isdir(annotation_folder):
        raise ValueError("Unable to locate ANNOTATIONS folder in the sample folder "+str(sample_path))
    prospective1 = os.path.join(annotation_folder,image_name+'_'+str(custom_label)+'.tif')
    outputs = []
    if not os.path.exists(prospective1):
        raise ValueError("Required custom annotation file is not present. "+str(prospective1))
    d1 =  _generate_file_dictionary(prospective1)
    d1['mask_label'] = custom_label
    outputs += [d1]
    return outputs

def _generate_file_dictionary(file_path):
    fileStatsObj = os.stat(file_path)
    modificationTime = time.ctime(fileStatsObj[stat.ST_MTIME])
    d = {
            'file_path': file_path,
            'sha256_hash':_sha256(file_path),
            'last_modified_timestamp':modificationTime
    }
    return d