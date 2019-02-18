import os, json
import pythologist
import pandas as pd
import numpy as np
from collections import OrderedDict

_inform_prefixes = ['First','Second','Third','Fourth','Fifth','Sixth','Seventh']

def write_inForm(idf,output_directory,overwrite=False):
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
            base = frame
            type = 'single-stain'
            if len(idf.scored_stains)>1:
                type = 'multi-stain'
                _make_score_file_multi(opath,sample,frame,base,fdf)
            elif len(idf.scored_stains)==1:
                _make_score_file_single(opath,sample,frame,base,fdf)
            else: raise ValueError("don't know how to do this with zero scored stains yet")
            _make_segmentation_file(opath,sample,frame,base,fdf,type)
            _make_summary_file(opath,sample,frame,base,fdf,type)

def _make_summary_file(opath,sample,frame,base,fdf,type):
    first = fdf['compartment_values'].iloc[0]
    stains = list(first.keys())
    compartments = [list(first[stain].keys()) for stain in first][0]

    first = fdf.iloc[0]
    td = json.loads(first['tissues_present'])
    tissues = list(json.loads(first['tissues_present']).keys())
    phenotypes = json.loads(first['phenotypes_present'])

    rows = []
    for tissue in tissues+['All']:
        for phenotype in phenotypes+['All']:
            mytissue = np.array([True]*fdf.shape[0])
            if tissue != 'All': mytissue = fdf['tissue']==tissue
            mypheno = np.array([True]*fdf.shape[0])
            if phenotype != 'All': mypheno = fdf['phenotype']==phenotype
            specific = fdf[mytissue&mypheno]
            o = OrderedDict()
            o['Path'] = opath
            o['Sample Name'] = sample
            o['Tissue Category'] = tissue
            o['Phenotype'] = phenotype
            o['Cell ID'] = 'all'
            o['Total Cells'] = specific.shape[0]
            if tissue == 'All':
                tpx = 0
                if o['Total Cells'] > 0: tpx = fdf['total_area'].iloc[0]
                o['Tissue Category Area (pixels)'] = int(tpx)
                o['Cell Density (per megapixel)'] = 0
                if tpx > 0: o['Cell Density (per megapixel)'] = 1000000*specific.shape[0]/fdf['total_area'].iloc[0]
            else:
                tpx = 0
                if o['Total Cells'] > 0: tpx = td[tissue]
                o['Tissue Category Area (pixels)'] = int(tpx)
                o['Cell Density (per megapixel)'] = 0
                if tpx > 0: o['Cell Density (per megapixel)'] = 1000000*specific.shape[0]/td[tissue]
            o['Cell X Position'] = ''
            o['Cell Y Position'] = ''
            o['Process Region ID'] = ''
            o['Distance from Process Region Edge (pixels)'] = ''
            o['Category Region ID'] = ''
            o['Distance from Tissue Category Edge (pixels)'] = ''



            for compartment in compartments:
                o[compartment+' Area (pixels)'] = 0
                o[compartment+' Area (percent)'] = 0
                o[compartment+' Compactness'] = ''
                o[compartment+' Minor Axis'] = ''
                o[compartment+' Major Axis'] = ''
                o[compartment+' Axis Ratio'] = ''
                o[compartment+' Axis Ratio'] = ''
                for stain in stains:
                    o[compartment+' '+stain+' Min (Normalized Counts, Total Weighting)'] = 0
                    o[compartment+' '+stain+' Mean (Normalized Counts, Total Weighting)'] = 0
                    o[compartment+' '+stain+' Max (Normalized Counts, Total Weighting)'] = 0
                    o[compartment+' '+stain+' Std Dev (Normalized Counts, Total Weighting)'] = 0
                    o[compartment+' '+stain+' Total (Normalized Counts, Total Weighting)'] = 0
            o['Entire Cell Area (pixels)'] = 0
            o['Entire Cell Area (percent)'] = 0
            o['Entire Cell Compactness'] = ''
            o['Entire Cell Minor Axis'] = ''
            o['Entire Cell Major Axis'] = ''
            o['Entire Cell Axis Ratio'] = ''
            o['Entire Cell Axis Ratio'] = ''
            for stain in stains:
                o['Entire Cell '+stain+' Min (Normalized Counts, Total Weighting)'] = 0
                o['Entire Cell '+stain+' Mean (Normalized Counts, Total Weighting)'] = 0
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

            rows.append(pd.Series(o))
    summary = pd.DataFrame(rows)
    if type == 'single-stain':
        summary = summary[summary['Tissue Category']=='All'].copy()
        summary = summary.drop(columns=['Tissue Category','Tissue Category Area (pixels)'])
    summary_file = os.path.join(opath,base+'_cell_seg_data_summary.txt')
    summary.to_csv(summary_file,sep="\t",index=False)



def _make_segmentation_file(opath,sample,frame,base,fdf,type):
    first = fdf['compartment_values'].iloc[0]
    stains = list(first.keys())
    compartments = [list(first[stain].keys()) for stain in first][0]
    segments = []
    for row in fdf.itertuples(index=False):
        s = pd.Series(row,index=fdf.columns)
        o = OrderedDict()
        o['Path'] = opath
        o['Sample Name'] = sample
        if type != "single-stain": o['Tissue Category'] = s['tissue']
        o['Phenotype'] = s['phenotype']
        o['Cell ID'] = s['id']
        o['Total Cells'] = ''
        if type != "single-stain": o['Tissue Category Area (pixels)'] = ''
        o['Cell Density (per megapixel)'] = ''
        o['Cell X Position'] = s['x']
        o['Cell Y Position'] = s['y']
        o['Process Region ID'] = '#N/A'
        o['Distance from Process Region Edge (pixels)'] = '#N/A'
        o['Category Region ID'] = 1
        o['Distance from Tissue Category Edge (pixels)'] = 0 #filler
        d = s['compartment_values']
        d1 = s['compartment_areas']
        for compartment in compartments:
            o[compartment+' Area (pixels)'] = d1[compartment]
            o[compartment+' Area (percent)'] = 0
            o[compartment+' Compactness'] = 0
            o[compartment+' Minor Axis'] = 0
            o[compartment+' Major Axis'] = 0
            o[compartment+' Axis Ratio'] = 0
            o[compartment+' Axis Ratio'] = 0
            for stain in stains:
                stats = ['Min','Mean','Max','Std Dev','Total']
                for stat in stats:
                    o[compartment+' '+stain+' '+stat+' (Normalized Counts, Total Weighting)'] = d[stain][compartment][stat]
        o['Entire Cell Area (pixels)'] = s['cell_area']
        o['Entire Cell Area (percent)'] = 0
        o['Entire Cell Compactness'] = 0
        o['Entire Cell Minor Axis'] = 0
        o['Entire Cell Major Axis'] = 0
        o['Entire Cell Axis Ratio'] = 0
        o['Entire Cell Axis Ratio'] = 0
        for stain in stains:
            stats = ['Min','Mean','Max','Std Dev','Total']
            for stat in stats:
                o['Entire Cell '+stain+' '+stat+' (Normalized Counts, Total Weighting)'] = s['entire_cell_values'][stain][stat]
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

def _make_score_file_multi(opath,sample,frame,base,fdf):
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
            prefix = _inform_prefixes[i]
            s = pd.Series(row,tdf.columns)
            o[prefix+' Cell Compartment'] = s['compartment']
            o[prefix+' Stain Component'] = s['stain']
        o['Double Negative'] = 0
        for i,row in enumerate(tdf.itertuples(index=False)):
            prefix = _inform_prefixes[i]
            s = pd.Series(row,tdf.columns)
            o['Single '+s['stain']] = 0
        o['Double Positive'] = 0
        o['Tissue Category Area (Percent)'] = 0
        o['Number of Cells'] = fdf.shape[0]
        for i,row in enumerate(tdf.itertuples(index=False)):
            prefix = _inform_prefixes[i]
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
def _make_score_file_single(opath,sample,frame,base,fdf):
    ### Do the score file
    score_formated = []
    score_df = fdf.score_data
    for tissue in score_df['tissue'].unique():
        tdf = score_df[score_df['tissue']==tissue]
        if tissue == 'any': tissue = ''
        if tdf.shape[0] > 1: raise ValueError("cant output multiple stains as single")
        o = OrderedDict()
        o['Path'] = opath
        o['Sample Name'] = sample
        o['Tissue Category'] = tissue
        o['Cell Compartment'] = tdf.iloc[0]['compartment']
        o['Stain Component'] = tdf.iloc[0]['stain']
        o['Positivity'] = 0
        o['Tissue Category Area (Percent)'] = 0
        o['Number of Cells'] = fdf.shape[0]
        o['Positivity Threshold'] = tdf.iloc[0]['threshold']
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
