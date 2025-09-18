#!/usr/bin/env python3
"""
Lunaphore Validation Tool with GUI
September 17, 2025
Modified from original by Ian Dryg
Dana-Farber Cancer Institute
Center for Immuno-Oncology

This code is used to VALIDATE outputs from Lunaphore's Horizon analysis software for use with pythologist. 
It quickly checks the output from Horizon and flags any issues before we proceed to data ingestion. 

GUI version allows user to select input file and output directory through file browser.
"""

import pandas as pd
import numpy as np
import os
import warnings
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import sys
from pathlib import Path

# Set pandas display options
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', None)


class LunaphoreValidatorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Lunaphore Validator")
        self.root.geometry("600x400")
        
        # Variables to store file paths
        self.input_file = tk.StringVar()
        self.output_dir = tk.StringVar()
        self.microns_per_pixel = tk.DoubleVar(value=0.28)
        
        self.setup_gui()
        
    def setup_gui(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Title
        title_label = ttk.Label(main_frame, text="Lunaphore Horizon Export Validator", 
                               font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # Input file selection
        ttk.Label(main_frame, text="Input File:").grid(row=1, column=0, sticky=tk.W, pady=5)
        ttk.Entry(main_frame, textvariable=self.input_file, width=50).grid(row=1, column=1, padx=5, pady=5)
        ttk.Button(main_frame, text="Browse", command=self.select_input_file).grid(row=1, column=2, padx=5, pady=5)
        
        # Output directory selection
        ttk.Label(main_frame, text="Output Directory:").grid(row=2, column=0, sticky=tk.W, pady=5)
        ttk.Entry(main_frame, textvariable=self.output_dir, width=50).grid(row=2, column=1, padx=5, pady=5)
        ttk.Button(main_frame, text="Browse", command=self.select_output_dir).grid(row=2, column=2, padx=5, pady=5)
        
        # Microns per pixel setting
        ttk.Label(main_frame, text="Microns per Pixel:").grid(row=3, column=0, sticky=tk.W, pady=5)
        ttk.Entry(main_frame, textvariable=self.microns_per_pixel, width=10).grid(row=3, column=1, sticky=tk.W, padx=5, pady=5)
        
        # Run button
        run_button = ttk.Button(main_frame, text="Run Validation", command=self.run_validation)
        run_button.grid(row=4, column=0, columnspan=3, pady=20)
        
        # Progress bar
        self.progress = ttk.Progressbar(main_frame, mode='indeterminate')
        self.progress.grid(row=5, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        
        # Output text area
        text_frame = ttk.Frame(main_frame)
        text_frame.grid(row=6, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=10)
        
        self.output_text = tk.Text(text_frame, height=15, width=70)
        scrollbar = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=self.output_text.yview)
        self.output_text.configure(yscrollcommand=scrollbar.set)
        
        self.output_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(6, weight=1)
        text_frame.columnconfigure(0, weight=1)
        text_frame.rowconfigure(0, weight=1)
        
    def select_input_file(self):
        filename = filedialog.askopenfilename(
            title="Select Horizon Export File",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if filename:
            self.input_file.set(filename)
            
    def select_output_dir(self):
        directory = filedialog.askdirectory(title="Select Output Directory")
        if directory:
            self.output_dir.set(directory)
            
    def log_message(self, message):
        """Add message to the output text area"""
        self.output_text.insert(tk.END, message + "\n")
        self.output_text.see(tk.END)
        self.root.update()
        
    def run_validation(self):
        # Validate inputs
        if not self.input_file.get():
            messagebox.showerror("Error", "Please select an input file")
            return
            
        if not self.output_dir.get():
            messagebox.showerror("Error", "Please select an output directory")
            return
            
        # Clear output text
        self.output_text.delete(1.0, tk.END)
        
        # Start progress bar
        self.progress.start()
        
        try:
            # Create output filename
            input_path = Path(self.input_file.get())
            output_filename = input_path.stem + "_validate.txt"
            output_path = Path(self.output_dir.get()) / output_filename
            
            self.log_message(f"Starting validation of: {input_path.name}")
            self.log_message(f"Output will be saved to: {output_path}")
            
            # Redirect stdout to capture print statements
            original_stdout = sys.stdout
            sys.stdout = self
            
            # Run the validation
            validate_lunaphore(
                str(input_path),
                str(self.output_dir.get()),
                self.microns_per_pixel.get()
            )
            
            # Restore stdout
            sys.stdout = original_stdout
            
            # Save output to file
            with open(output_path, 'w') as f:
                f.write(self.output_text.get(1.0, tk.END))
            
            self.log_message(f"\nValidation complete! Results saved to: {output_path}")
            messagebox.showinfo("Success", f"Validation complete!\nResults saved to: {output_path}")
            
        except Exception as e:
            sys.stdout = original_stdout
            error_msg = f"Error during validation: {str(e)}"
            self.log_message(error_msg)
            messagebox.showerror("Error", error_msg)
        
        finally:
            self.progress.stop()
    
    def write(self, text):
        """Method to capture stdout and redirect to GUI"""
        if text.strip():  # Only log non-empty strings
            self.log_message(text.strip())
    
    def flush(self):
        """Required for stdout redirection"""
        pass


# Your original validation functions (unchanged)
def validate_lunaphore(horizon_export_filepath, savefile_dir, microns_per_pixel=0.28):
    # validate input parameters
    try:
        validate_parameters(horizon_export_filepath, savefile_dir, microns_per_pixel)
    except (TypeError, ValueError) as e:
        print(f"Error: {e}")
        return
    
    print("---------------------------------------------------------")
    print("---------------------------------------------------------")
    print(str(horizon_export_filepath))
    print("---------------------------------------------------------")
    print("---------------------------------------------------------")

    # read in the horizon output
    temp_input_df1 = pd.read_csv(horizon_export_filepath)

    # remove unwanted column
    if 'Unnamed: 0' in list(temp_input_df1.columns):
        temp_input_df1 = temp_input_df1.drop(columns=['Unnamed: 0'])
        
    # cell annotations have Nuclei Segmentation at the end of the Annotation Group "path", label those rows
    temp_input_df1.loc[temp_input_df1['Annotation Group'].str.contains('Nuclei Segmentation'), 'annot_name'] = 'Nuclei Segmentation'
    # area annotations have the annotation name at the end of the Annotation Group "path". Extract those names
    temp_input_df1.loc[~temp_input_df1['Annotation Group'].str.contains('Nuclei Segmentation'), 'annot_name'] = temp_input_df1['Annotation Group'].str.split('/').str[-1]
    # Split the cells and the areas dfs apart. 
    # cells df has Nuclei Segmentation in the Annotation Group "path", extract those rows
    temp_input_cells = temp_input_df1.loc[temp_input_df1['Annotation Group'].str.contains('Nuclei Segmentation')]
    # area df does not have Nuclei Segmentation in the Annotation Group "path", extract those rows
    temp_input_area = temp_input_df1.loc[~temp_input_df1['Annotation Group'].str.contains('Nuclei Segmentation')]
    
    # remove Exclusions from cells dataframe if there are any
    if 'Class group' in list(temp_input_cells.columns):
        print('Removing exclusions...')
        print('Size before removing exclusions: ' + str(temp_input_cells.shape))
        temp_input_cells = temp_input_cells.loc[temp_input_cells['Class group']!='Exclusions']
        print('Size after removing exclusions: ' + str(temp_input_cells.shape))

    # Label parent annotations for CELLS (Main and ROI). Should be in Class group column. 
    temp_input_cells['parent_id'] = temp_input_cells['Class group'].str.lower().str.split('_').str[1]
    temp_input_cells['parent_main_id'] = temp_input_cells['Class group'].str.lower().str.split('_').str[1].str.split('.').str[0]
    temp_input_cells['parent_roi_id'] = temp_input_cells['Class group'].str.lower().str.split('_').str[1].str.split('.').str[1]
    
    # drop columns that only contain nan
    temp_input_cells = temp_input_cells.dropna(axis=1, how='all')
    temp_input_area = temp_input_area.dropna(axis=1, how='all')

    # Label annotation info for all annotations
    # ex: ROI_1.1.0_Polygon
    temp_input_area['annot_type'] = temp_input_area['annot_name'].str.lower().str.split('_').str[0]
    temp_input_area['annot_shape'] = temp_input_area['annot_name'].str.lower().str.split('_').str[2]
    temp_input_area['full_annot_id'] = temp_input_area['annot_name'].str.lower().str.split('_').str[1]
    temp_input_area['main_annot_id'] = temp_input_area['annot_name'].str.lower().str.split('_').str[1].str.split('.').str[0]
    temp_input_area['roi_annot_id'] = temp_input_area['annot_name'].str.lower().str.split('_').str[1].str.split('.').str[1]
    temp_input_area['exclusion_annot_id'] = temp_input_area['annot_name'].str.lower().str.split('_').str[1].str.split('.').str[2]

    print('area df: ')
    print(temp_input_area.to_string())

    # ---------------------------------------------------------
    print("---------------------------------------------------------")
    print("---------------------------------------------------------")
    print("Check cells (excluded cells were dropped already):")
    print(temp_input_cells['Class group'].value_counts())
    print("---------------------------------------------------------")
    print("---------------------------------------------------------")
    print("Review area annotations:")
    print(temp_input_area['full_annot_id'].to_string())
    print("---------------------------------------------------------")
    print("---------------------------------------------------------")
    print("Check annotations for required columns:")
    areas_req_cols = ['Annotation Group','Area in μm²','X Position in μm','Y Position in μm']
    missing_area_cols = [x for x in areas_req_cols if x not in temp_input_area.columns]
    if len(missing_area_cols) > 0:
        print('Annotations are missing the following columns:')
        print(missing_area_cols)
    print("---------------------------------------------------------")
    print("---------------------------------------------------------")
    print("Check if annotations are missing fields: ")
    print("Number of NaNs in required columns: ")
    for curr_col in areas_req_cols:
        if curr_col in temp_input_area.columns:
            print(curr_col)
            num_nans = temp_input_area[curr_col].isna().sum()
            print(str(num_nans))
            if num_nans > 0:
                print("Annotations with NaNs: ")
                print(temp_input_area.loc[temp_input_area[curr_col].isna()]['Annotation Group'].to_string())
    print("---------------------------------------------------------")
    print("---------------------------------------------------------")
    print("view areas df: ")
    print(temp_input_area.to_string())
    print("---------------------------------------------------------")
    print("---------------------------------------------------------")
    print("Check cells for duplicate thresholds: ")
    # ---------------------------------------------------------
    # check meta on cells data, drop duplicate markers if necessary
    qc_input_cells = extract_column_metadata(temp_input_cells)
    # check for duplicate thresholds per marker in the cell compartment
    qc_input_cells_cell_thresholds = qc_input_cells.loc[(qc_input_cells['Measurement_Type']=='Threshold') & (qc_input_cells['Compartment_Type']=='Cell')]
    # keep duplicates AND originals to investigate
    qc_input_cells_threshold_duplicates = qc_input_cells_cell_thresholds.loc[qc_input_cells_cell_thresholds.duplicated(subset=['Label_Mapping'], keep=False)]
    # Warning and display if there are ny duplicates. 
    if not qc_input_cells_threshold_duplicates.empty:
        unique_labels = qc_input_cells_threshold_duplicates['Label_Mapping'].unique()
        warnings.warn(
            f"---------------------------------------------------------\n"
            f"Duplicate thresholds found for the following markers: \n"
            f"Duplicated Label_Mapping values: {list(unique_labels)}\n"
            f"---------------------------------------------------------\n"
        )
    # ---------------------------------------------------------
    print("---------------------------------------------------------")
    print("---------------------------------------------------------")


def validate_parameters(horizon_export_filepath, savefile_dir, microns_per_pixel):
    # Type checking
    if not isinstance(horizon_export_filepath, str):
        raise TypeError("horizon_export_filepath must be a string")
    if not isinstance(savefile_dir, str):
        raise TypeError("savefile_dir must be a string")
    if not isinstance(microns_per_pixel, float):
        raise TypeError("microns_per_pixel must be a float")
    # validate files exist
    if not os.path.isfile(horizon_export_filepath):
        raise ValueError(f"The path '{horizon_export_filepath}' is not a file")


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
        'region_label':'region_label',
        'parent_id':'Parent Annotation',
        'Leiden clusters':'Leiden clusters'
    }
    
    compartment_type_map = {
        'Cells':'Cell',
        'Cytoplasm':'Cytoplasm',
        'Nuclei':'Nucleus',
        'Annotation':'Annotation',
        'Position':'Annotation',
        'region':'Annotation',
        'parent_id':'Annotation',
        'Leiden clusters':'Annotation'
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
    temp_thresholds_df = df['orig_cols'].str.split('Threshold \\(', expand=True)
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


def check_markers(meta):
    print(list(meta['Marker'].dropna().unique()))


def rename_marker(meta, old_marker_name, new_marker_name):
    meta['Marker'] = meta['Marker'].replace(old_marker_name, new_marker_name)
    # also update Label_Mapping column
    meta['Label_Mapping'] = meta['Label_Mapping'].replace(old_marker_name, new_marker_name)
    return meta


def main():
    """Main function to launch the GUI"""
    root = tk.Tk()
    app = LunaphoreValidatorGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()