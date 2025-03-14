# December 20, 2024
# Ian Dryg
# Dana-Farber Cancer Institute
# Center for Immuno-Oncology
# This code is used to ingest outputs from Lunaphore's Horizon analysis software for use with pythologist. 
# It converts the output from Horizon into a pythologist CellDataFrame for further use in our pipelines. 

# update: December 27, 2024
# Harry's Horizon outputs had area and cells dataframes combined in one file. 
# The area data is the first row(s). 
# Then the cells data is after that. 
# Some columns are only used by one or the other. 
# Let's make a function to optionally take the one file version as input and split them into cells and area dataframes for further processing. 

import pandas as pd
import numpy as np
import uuid
from pythologist import CellDataFrame
import os

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', None)


# run all the ingestion methods
# cells_df_path: path to cells csv file. Example: 'data/CELLS_20241125.csv'
# area_df_path: path to areas csv file. Example: 'data/AREA_20241125.csv'
# Note: in this version, if the cells_df_path and area_df_path are the same, the program assumes they're combined in the same file and will split them. 
# project_name: name of the project. Example: 'Nick_Horizon_Testing_20241219'
# microns_per_pixel: should be 0.28 for the Lunaphore instrument as of 20241220
# run_qc: set to True if you want to check the cdf qc
# return_cdf: set to True if you want to return the cdf, otherwise just saves the file and doesn't return anything
def run_lunaphore_ingestion(cells_df_path,
                            area_df_path,
                            project_name,
                            savefile_path,
                            overwrite_sample_name = None,
                            default_phenotype = 'CD3', 
                            microns_per_pixel=0.28,
                            run_qc = False,
                            save_cdf = False,
                            return_cdf = False,
                            show_meta = False,
                            rename_markers_dict = None):
    
    # validate input parameters
    try:
        validate_parameters(cells_df_path, area_df_path, project_name, savefile_path, microns_per_pixel, run_qc, return_cdf, overwrite_sample_name, default_phenotype)
    except (TypeError, ValueError) as e:
        print(f"Error: {e}")
        return
    
    # If the cells_df_path and area_df_path are the same, assume that file contains both area and cells data. 
    # In that case, we need to split them apart. 
    if cells_df_path != area_df_path:
        # read in csv files separately
        cells = pd.read_csv(cells_df_path)
        area = pd.read_csv(area_df_path)
        # check for Unnamed: 0 column
        if 'Unnamed: 0' in list(cells.columns): 
            cells = cells.drop(columns=['Unnamed: 0'])
    if cells_df_path == area_df_path:
        # assume info is in one file
        temp_input_df1 = pd.read_csv(cells_df_path)
        # cells df has Nuclei Segmentation in the Annotation Group "path", extract those rows
        temp_input_cells = temp_input_df1.loc[temp_input_df1['Annotation Group'].str.contains('Nuclei Segmentation')]
        # area df does not have Nuclei Segmentation in the Annotation Group "path", extract those rows
        temp_input_area = temp_input_df1.loc[~temp_input_df1['Annotation Group'].str.contains('Nuclei Segmentation')]
        # drop columns that only contain nan
        cells = temp_input_cells.dropna(axis=1, how='all')
        area = temp_input_area.dropna(axis=1, how='all')

    
    # add region areas
    cells = add_region_areas(cells, area, microns_per_pixel)
    
    # extract column metadata
    meta = extract_column_metadata(cells)
    
    # rename markers list if specified
    if rename_markers_dict is not None:
        for old_marker_name, new_marker_name in rename_markers_dict.items():
            meta = rename_marker(meta, old_marker_name, new_marker_name)
    
    if show_meta:
        display(meta)
    
    # convert to pythologist CellDataFrame
    cdf = ingest_Lunaphore(cells, 
                           meta, 
                           proj_name=project_name, 
                           default_phenotype=default_phenotype, 
                           microns_per_pixel=0.28,
                           overwrite_sample_name = overwrite_sample_name)
    
    if save_cdf:
        # save cdf, as .cdf.h5
        # example savefile_path: 'Processing/Nick_Horizon_Testing_20241219.cdf.h5'
        cdf.to_hdf(savefile_path,'data')
    
    if run_qc:
        qc = cdf.qc()
        qc.run_tests()
        qc.print_results()
    
    if return_cdf:
        return cdf


# function to validate input parameters before proceeding
def validate_parameters(cells_df_path, area_df_path, project_name, savefile_path, microns_per_pixel, run_qc, return_cdf, overwrite_sample_name, default_phenotype):
    # Type checking
    if not isinstance(cells_df_path, str):
        raise TypeError("cells_df_path must be a string")
    if not isinstance(area_df_path, str):
        raise TypeError("area_df_path must be a string")
    if not isinstance(project_name, str):
        raise TypeError("project_name must be a string")
    if not isinstance(savefile_path, str):
        raise TypeError("savefile_path must be a string")
    if not isinstance(microns_per_pixel, float):
        raise TypeError("microns_per_pixel must be a float")
    if not isinstance(run_qc, bool):
        raise TypeError("run_qc must be a boolean")
    if not isinstance(return_cdf, bool):
        raise TypeError("return_cdf must be a boolean")
    if overwrite_sample_name is not None:
        if not isinstance(overwrite_sample_name, str):
                raise TypeError("overwrite_sample_name must be a string")
    if not isinstance(default_phenotype, str):
        raise TypeError("default_phenotype must be a string")
    # validate files exist
    if not os.path.isfile(cells_df_path):
        raise ValueError(f"The path '{cells_df_path}' is not a file")
    if not os.path.isfile(area_df_path):
        raise ValueError(f"The path '{area_df_path}' is not a file")


# function to integrate region areas into the cells dataframe. 
# regions will just be called ANY (assuming there's only one)
def add_region_areas(cells_df, area_df, microns_per_pixel):
    # Set region label to ANY
    area_df['region_label'] = 'ANY'
    cells_df['region_label'] = 'ANY'
    # convert areas from microns into pixels, which is expected by pythologst. 
    # area_pixles2 = area_microns2*((1/microns_per_pixel)**2)
    area_df['Area_pixels_squared'] = area_df['Area in μm²'].apply(lambda x: round(x*((1/microns_per_pixel)**2), 3))
    # Extract dictionary of region areas using ANY as region label
    region_areas_dict = dict(zip(area_df['region_label'],area_df['Area_pixels_squared']))
    # put this region areas dict into the regions column. It'll be the same value for every row. 
    cells_df['regions'] = cells_df['region_label'].apply(lambda x: region_areas_dict)
    
    return cells_df


# function to integrate region areas into the cells dataframe. 
def old_add_region_areas(cells_df, area_df):
    # Want to drop the Nuclei Segmentation part of the Annotation Group. 
    # explode Annotation Group
    temp_region_df = cells_df['Annotation Group'].str.split('/', expand=True)
    # get number of columns 
    num_cols = len(list(temp_region_df.columns))
    # keep all but last one and put them back together
    temp_region_df = temp_region_df.iloc[:,0:num_cols-1]
    # Recombine back together
    cells_df['region_label'] = temp_region_df.apply(lambda x: '/'.join(x), axis=1)
    # Extract dictionary of region areas. 
    region_areas_dict = dict(zip(area_df['Annotation Group'],area_df['Area in μm²']))
    # put this region areas dict into the regions column. It'll be the same value for every row. 
    cells_df['regions'] = cells_df['region_label'].apply(lambda x: region_areas_dict)
    
    return cells_df


# function to extract metadata
# This function labels columns in the dataset to indicate which columns should be used for what later on. 
# It also helps set up to rename the columns later. 

def extract_column_metadata(cells_df):
    import numpy as np
    
    # extract columns
    df = pd.DataFrame({'orig_cols':cells_df.columns})
    
    # define key measurements to reformat
    measurement_type_map = {
        'Annotation Group':'Annotation Group',
        'Annotation Index':'cell_index',
        'X Position':'x',
        'Y Position':'y',
        'Area':'cell_area',
        'Mean Intensity':'Mean Intensity',
        'Threshold':'Threshold',
        'regions':'regions',
        'region_label':'region_label'
    }
    
    compartment_type_map = {
        'Cells':'Cell',
        'Cytoplasm':'Cytoplasm',
        'Nuclei':'Nucleus',
        'Annotation':'Annotation',
        'Position':'Annotation',
        'region':'Annotation'
    }
    
    # ----------------------------
    # Extract Measurement Type
    df['Measurement_Type'] = ''
    for k,v in measurement_type_map.items():
        df.loc[df['orig_cols'].str.contains(k), 'Measurement_Type'] = v
    
    # ----------------------------
    # Extract Compartment Type
    df['Compartment_Type'] = ''
    for k,v in compartment_type_map.items():
        df.loc[df['orig_cols'].str.contains(k), 'Compartment_Type'] = v
    
    # ----------------------------
    # Extract Marker Label
    temp_marker_df = df['orig_cols'].str.split('(', expand=True)
    # use  3rd column of output to separate further, rows without values should be None...
    temp_marker_df2 = temp_marker_df.iloc[:, 2].str.split(' - ', expand=True)
    # Nucleus has slightly different structure. Marker is right after first (, not second. 
    # use  2nd column of output to separate further, rows without values should be None...
    temp_marker_df2_nucleus = temp_marker_df.iloc[:, 1].str.split(' - ', expand=True)
    # conditionally fill where it's Nucleus and Mean Intensity using nucleus outputs, otherwise use other outputs. 
    # use 1st column to separate out the marker
    df['Marker'] = np.where((df['Compartment_Type']=='Nucleus') & (df['Measurement_Type']=='Threshold'), 
                            temp_marker_df2_nucleus.iloc[:, 0], 
                            temp_marker_df2.iloc[:, 0])
    # manually edit DAPI
    df.loc[df['orig_cols'].str.contains('DAPI'),'Marker'] = 'DAPI'
    # get just the first part of Marker...
    # some don't already have _ so we add one just in case
    df['Marker'] = df['Marker'] + '_'
    temp_marker_df_first = df['Marker'].str.split('_', expand=True)
    df['Marker'] = temp_marker_df_first.iloc[:, 0]
    # Remove hyphens
    df['Marker'] = df['Marker'].str.replace('-', '')
    
    # ----------------------------
    # Extract Channel Label
    # use temp_marker_df2 for most fields
    temp_marker_df3 = temp_marker_df2.iloc[:, 1].str.split(')', expand=True)
    # use temp_marker_df2_nucleus for nuclei
    if temp_marker_df2_nucleus.shape[1] > 1:
        # if there are nuclei...
        temp_marker_df3_nucleus = temp_marker_df2_nucleus.iloc[:, 1].str.split(')', expand=True)
    else:
        # if there aren't nuclei... (turn series into dataframe)
        temp_marker_df3_nucleus = pd.DataFrame(temp_marker_df2_nucleus.iloc[:,0].apply(lambda x: np.nan))
    # conditionally fill where it's Nucleus and Mean Intensity
    df['Channel'] = np.where((df['Compartment_Type']=='Nucleus') & (df['Measurement_Type']=='Threshold'),
                             temp_marker_df3_nucleus.iloc[:, 0],
                             temp_marker_df3.iloc[:, 0])
    # manually edit DAPI
    df.loc[df['orig_cols'].str.contains('DAPI'),'Channel'] = 'DAPI'
    
    # ----------------------------
    # Extract Thresholds
    # same for nuclei and other cell compartments
    temp_thresholds_df = df['orig_cols'].str.split('Threshold \(', expand=True)
    # Take 2nd column, after the "Threshold ("
    temp_thresholds_df2 = temp_thresholds_df.iloc[:, 1].str.split(')', expand=True)
    # Take 1st column, before the ")"
    df['Threshold'] = temp_thresholds_df2.iloc[:, 0]
    
    # ----------------------------
    # Edge cases:
    # manually set the Measurement Type for Area Threshold and Position Thresholds
    df.loc[(df['orig_cols'].str.contains('Area')) & (df['orig_cols'].str.contains('Threshold')),'Measurement_Type'] = 'Area Threshold'
    df.loc[(df['orig_cols'].str.contains('Position')) & (df['orig_cols'].str.contains('Threshold')),'Measurement_Type'] = 'Position Threshold'
    # By default let's use x and y positions of Cell (should be same as nuclei anyway)
    # so let's set Measurement_Type of the nucelei x and y to something specific...
    df.loc[(df['orig_cols'].str.contains('X Position')) & (df['orig_cols'].str.contains('Nuclei')),'Measurement_Type'] = 'nucleus_x'
    df.loc[(df['orig_cols'].str.contains('Y Position')) & (df['orig_cols'].str.contains('Nuclei')),'Measurement_Type'] = 'nucleus_y'
    # manually set Marker name to nan for these
    df.loc[(df['orig_cols'].str.contains('Area')) & (df['orig_cols'].str.contains('Threshold')),'Marker'] = np.nan
    df.loc[(df['orig_cols'].str.contains('Position')) & (df['orig_cols'].str.contains('Threshold')),'Marker'] = np.nan
    
    # ----------------------------
    # Make a column defining renaming:
    # use markers 
    df['Label_Mapping'] = df['Marker']
    # if markers are na, use measurement type
    df.loc[df['Label_Mapping'].isna(), 'Label_Mapping'] = df['Measurement_Type']
    
    # return column metadata
    return df


# function to check Marker names
def check_markers(meta):
    display(list(meta['Marker'].dropna().unique()))


# function to rename Marker names
def rename_marker(meta, old_marker_name, new_marker_name):
    meta['Marker'] = meta['Marker'].replace(old_marker_name, new_marker_name)
    # also update Label_Mapping column
    meta['Label_Mapping'] = meta['Label_Mapping'].replace(old_marker_name, new_marker_name)
    return meta


# function to transform Lunaphore Horizon output data into a pythologist CellDataFrame
def ingest_Lunaphore(df, 
                     meta, 
                     proj_name, 
                     default_phenotype='CD3', 
                     microns_per_pixel=0.28,
                     overwrite_sample_name = None):
    import numpy as np
    import uuid
    
    # drop Area Threshold and Position Threshold
    meta = meta.loc[~meta['Measurement_Type'].isin(['Area Threshold','Position Threshold'])]
    # drop DAPI
    meta = meta.loc[meta['Marker']!='DAPI']
    # drop nucleus_x and nucleus_y
    meta = meta.loc[~meta['Measurement_Type'].isin(['nucleus_x','nucleus_y'])]
    # drop Nucleus: Mean Intensity combination
    meta = meta.loc[~((meta['Compartment_Type']=='Nucleus') & (meta['Measurement_Type']=='Mean Intensity'))]
    
    # --------------------------------------------------------------------
    # Process Values
    # subset meta to name mappings for Cell (values)
    value_name_mappings = meta.loc[(meta['Compartment_Type'].isin(['Cell', 'Annotation'])) & (meta['Measurement_Type']!='Threshold')]
    
    # make column name remapping dict
    vals_remapping_dict = dict(zip(value_name_mappings['orig_cols'], value_name_mappings['Label_Mapping']))
    # get list of columns to keep 
    vals_cols_keep = list(value_name_mappings['orig_cols'])
    # subset data to include only values columns
    df_vals = df[vals_cols_keep]
    print('values columns subset')
    df_vals = df_vals.rename(columns = vals_remapping_dict)
    print('values columns renamed')
    
    # Use these columns for index
    index_columns = ['Annotation Group','cell_index','cell_area','x','y','region_label']
    # Pull out regions column since it's not hashable (can't set it to index)
    df_regions = df_vals[index_columns + ['regions']].set_index(index_columns)
    # drop regions column from df_vals because it's not hashable. Will merge it back in later. 
    df_vals = df_vals.drop(columns=['regions'])
    
    # Want to collapse columns into a dictionary column. 
    # Set non value columns to index. 
    df_vals = df_vals.set_index(index_columns)
    # round to four decimal places. 
    df_vals = df_vals.round(4)
    # make channel values column and drop the individual values columns
    leftover_cols = list(df_vals.columns)
    df_vals['channel_values'] = df_vals.apply(lambda row: row.to_dict(), axis=1)
    df_vals = df_vals.drop(columns=leftover_cols)
    
    # --------------------------------------------------------------------
    # Process Thresholds
    # subset meta to name mappings for Threshold (calls)
    call_name_mappings = meta.loc[(meta['Compartment_Type'].isin(['Cell', 'Nucleus', 'Annotation'])) & (meta['Measurement_Type']!='Mean Intensity')]
    # don't want to keep nucleus area...
    call_name_mappings = call_name_mappings.loc[~((call_name_mappings['Compartment_Type']=='Nucleus') & (call_name_mappings['Measurement_Type']=='cell_area'))]
    
    # make column name remapping dict
    calls_remapping_dict = dict(zip(call_name_mappings['orig_cols'], call_name_mappings['Label_Mapping']))
    # get list of columns to keep 
    calls_cols_keep = list(call_name_mappings['orig_cols'])
    # subset data to include only threshold columns
    df_calls = df[calls_cols_keep]
    print('calls columns subset')
    df_calls = df_calls.rename(columns = calls_remapping_dict)
    print('calls columns renamed')
    # drop regions column from df_calls because it's not hashable. Will merge it back in later. 
    df_calls = df_calls.drop(columns=['regions'])
    # Want to collapse columns into a dictionary column. 
    # Set non call columns to index. 
    df_calls = df_calls.set_index(index_columns)
    # pull out phenotype call
    if default_phenotype in list(df_calls.columns):
        # If the default phenotype is present, use that
        df_phenos = df_calls[[default_phenotype]]
    else:
        # if not, just use first scored call column
        # update default_phenotype to the first column
        default_phenotype = list(df_calls.columns)[0]
        df_phenos = df_calls[[default_phenotype]]
    # Drop that phenotype from the scored calls
    df_calls = df_calls.drop(columns=[default_phenotype])
    # Generate OTHER phenotype as the opposite boolean of the default phenotype
    df_phenos['OTHER'] = ~df_phenos[default_phenotype].astype(bool)
    # convert Booleans to 0 and 1s. For both scored calls and new phenotype
    df_calls = df_calls.astype(int)
    df_phenos = df_phenos.astype(int)
    # make scored calls column and drop the individual calls columns
    leftover_cols = list(df_calls.columns)
    df_calls['scored_calls'] = df_calls.apply(lambda row: row.to_dict(), axis=1)
    df_calls = df_calls.drop(columns=leftover_cols)
    # make phenotype calls column and drop the individual calls columns
    leftover_cols = list(df_phenos.columns)
    df_phenos['phenotype_calls'] = df_phenos.apply(lambda row: row.to_dict(), axis=1)
    # before dropping other calls, use the default phenotype column to create phenotype_label column
    df_phenos['phenotype_label'] = np.where(df_phenos[default_phenotype] == 1, default_phenotype, 'OTHER')
    # drop other calls columns
    df_phenos = df_phenos.drop(columns=leftover_cols)
    
    # merge calls and phenotypes
    df_phenos_calls = df_calls.merge(df_phenos, how='inner', left_index=True, right_index=True)
    
    # --------------------------------------------------------------------
    # Merge vals and calls back together on the indexes. 
    df_merge = df_vals.merge(df_phenos_calls, how='inner', left_index=True, right_index=True)
    # merge the regions column back in using indexes. 
    df_merge = df_merge.merge(df_regions, how='inner', left_index=True, right_index=True)
    #print('df_vals' + str(df_vals.shape))
    #print('df_phenos_calls' + str(df_phenos_calls.shape))
    #print('df_regions' + str(df_regions.shape))
    #print('df_merge' + str(df_merge.shape))
    #print('vals and calls merged')
    
    # Add some other pythologist columns
    # neighbors	frame_name	frame_id	sample_name	project_name	sample_id	project_id	frame_shape
    df_merge = df_merge.reset_index()
    df_merge['neighbors'] = np.nan
    df_merge['frame_id'] = uuid.uuid4().hex
    if overwrite_sample_name is not None:
        df_merge['sample_name'] = overwrite_sample_name
        df_merge['frame_name'] = overwrite_sample_name
    else:
        df_merge['sample_name'] = df_merge['region_label']
        df_merge['frame_name'] = df_merge['region_label']
    df_merge['project_name'] = proj_name
    df_merge['project_id'] = uuid.uuid4().hex
    df_merge['sample_id'] = uuid.uuid4().hex
    df_merge['frame_shape'] = df_merge['region_label'].apply(lambda x: tuple([1000,1000]))
    #print('other pythologist columns added')
    
    # convert to pythologist CellDataFrame
    cdf = CellDataFrame(df_merge)
    # 0.28 microns/pixel
    cdf.microns_per_pixel = microns_per_pixel
    
    return cdf