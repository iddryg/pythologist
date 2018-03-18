import os
import pythologist
import pandas as pd
from collections import OrderedDict

_prefixes = ['First','Second','Third','Fourth','Fifth','Sixth','Seventh']
_vectra_score_header = [
 'Path',
 'Sample Name',
 'Tissue Category'
 'First Cell Compartment',
 'First Stain Component',
 'Second Cell Compartment',
 'Second Stain Component',
 'Double Negative',
 'Single CD163 (Opal 690)',
 'Single PDL1 (Opal 520)',
 'Double Positive',
 'Tissue Category Area (Percent)',
 'Number of Cells',
 'CD163 (Opal 690) Threshold',
 'PDL1 (Opal 520) Threshold',
 'Lab ID',
 'Slide ID',
 'TMA Sector',
 'TMA Row',
 'TMA Column',
 'TMA Field',
 'inForm 2.3.6298.15583']

def write_inForm(idf,output_directory,type="Vectra",overwrite=False):
    if os.path.exists(output_directory) and overwrite is False:
        raise ValueError("Cannot create directory '"+output_directory+"'' since it already exists")
    for folder in idf['folder'].unique():
        opath = os.path.join(output_directory,folder)
        if not os.path.exists(opath): os.makedirs(opath)
        sub = idf[idf['folder']==folder]
        sample = sub['sample'].iloc[0]
        if sub['sample'].unique().shape[0] > 1: raise ValueError("more than one sample name present in folder "+folder)
        for frame in sub['frame'].unique():
            fdf = sub[sub['frame']==frame]
            fdf = pythologist.InFormCellFrame(fdf)
            base = sample+'_'+frame
            if frame.startswith(sample): base = frame
            _make_score_file(opath,sample,frame,base,fdf)
            _make_segmentation_file(opath,sample,frame,base,fdf)

def _make_segmentation_file(opath,sample,frame,base,fdf):
    first = fdf['compartment_values'].iloc[0]
    stains = list(first.keys())
    compartments = [list(first[stain].keys()) for stain in first][0]
    segments = []
    for row in fdf.itertuples(index=False):
        s = pd.Series(row,index=fdf.columns)
        o = OrderedDict()
        o['Path'] = opath
        o['Sample Name'] = sample
        o['Tissue Category'] = s['tissue']
        o['Phenotype'] = s['phenotype']
        o['Cell ID'] = s['id']
        o['Total Cells'] = ''
        o['Tissue Category Area (pixels)'] = ''
        o['Cell Density (per megapixel)'] = ''
        o['Cell X Position'] = s['x']
        o['Cell Y Position'] = s['y']
        o['Process Region ID'] = '#N/A'
        o['Distance from Process Region Edge (pixels)'] = '#N/A'
        o['Category Region ID'] = 1
        o['Distance from Tissue Category Edge (pixels)'] = 0 #filler
        d = s['compartment_values']
        for compartment in compartments:
            o[compartment+' Area (pixels)'] = 0
            o[compartment+' Area (percent)'] = 0
            o[compartment+' Compactness'] = 0
            o[compartment+' Minor Axis'] = 0
            o[compartment+' Major Axis'] = 0
            o[compartment+' Axis Ratio'] = 0
            o[compartment+' Axis Ratio'] = 0
            for stain in stains:
                o[compartment+' '+stain+' Min (Normalized Counts, Total Weighting)'] = 0
                o[compartment+' '+stain+' Mean (Normalized Counts, Total Weighting)'] = d[stain][compartment]
                o[compartment+' '+stain+' Max (Normalized Counts, Total Weighting)'] = 0
                o[compartment+' '+stain+' Std Dev (Normalized Counts, Total Weighting)'] = 0
                o[compartment+' '+stain+' Total (Normalized Counts, Total Weighting)'] = 0
        o['Entire Cell Area (pixels)'] = s['cell_area']
        o['Entire Cell Area (percent)'] = 0
        o['Entire Cell Compactness'] = 0
        o['Entire Cell Minor Axis'] = 0
        o['Entire Cell Major Axis'] = 0
        o['Entire Cell Axis Ratio'] = 0
        o['Entire Cell Axis Ratio'] = 0
        for stain in stains:
            o['Entire Cell '+stain+' Min (Normalized Counts, Total Weighting)'] = 0
            o['Entire Cell '+stain+' Mean (Normalized Counts, Total Weighting)'] = s['entire_cell_values'][stain]
            o['Entire Cell '+stain+' Max (Normalized Counts, Total Weighting)'] = 0
            o['Entire Cell '+stain+' Std Dev (Normalized Counts, Total Weighting)'] = 0
            o['Entire Cell '+stain+' Total (Normalized Counts, Total Weighting)'] = 0
        o['Lab ID'] = ''
        o['Slide ID'] = sample
        o['TMA Sector'] = 0
        o['TMA Row'] = 0
        o['TMA Column'] = 0
        o['TMA Field'] = 0
        o['Confidence'] = 0
        o['inForm 2.3.6298.15583'] = ''
        segments.append(pd.Series(o))
    segments = pd.DataFrame(segments)
    segments_file = os.path.join(opath,base+'_cell_seg_data.txt')
    segments.to_csv(segments_file,sep="\t",index=False)

def _make_score_file(opath,sample,frame,base,fdf):
    ### Do the score file
    score_formated = []
    score_df = fdf.score_data
    for tissue in score_df['tissue'].unique():
        tdf = score_df[score_df['tissue']==tissue]
        o = OrderedDict()
        o['Path'] = opath
        o['Sample Name'] = sample
        o['Tissue Category'] = tissue
        for i,row in enumerate(tdf.itertuples(index=False)):
            prefix = _prefixes[i]
            s = pd.Series(row,tdf.columns)
            o[prefix+' Cell Compartment'] = s['compartment']
            o[prefix+' Stain Component'] = s['stain']
        o['Double Negative'] = 0
        for i,row in enumerate(tdf.itertuples(index=False)):
            prefix = _prefixes[i]
            s = pd.Series(row,tdf.columns)
            o['Single '+s['stain']] = 0
        o['Double Positive'] = 0
        o['Tissue Category Area (Percent)'] = 0
        o['Number of Cells'] = fdf.shape[0]
        for i,row in enumerate(tdf.itertuples(index=False)):
            prefix = _prefixes[i]
            s = pd.Series(row,tdf.columns)
            o[s['stain']+' Threshold'] = s['threshold']
        o['Lab ID'] = ''
        o['Slide ID'] = sample
        o['TMA Sector'] = 0
        o['TMA Row'] = 0
        o['TMA Column'] = 0
        o['TMA Field'] = 0
        o['inForm 2.3.6298.15583'] = ''
        score_formated.append(pd.Series(o))
    score_formated = pd.DataFrame(score_formated)
    score_file = os.path.join(opath,base+'_score_data.txt')
    score_formated.to_csv(score_file,sep="\t",index=False)
