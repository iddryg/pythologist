from pythologist_reader.formats.inform.immunoprofile import CellSampleInFormImmunoProfile
from pythologist import CellDataFrame, SubsetLogic as SL, PercentageLogic as PL
import pandas as pd
import json, sys

def execute_immunoprofile_extraction(
    path,
    sample_name,
    panel_source,
    panel_name,
    panel_version,
    report_source,
    report_name,
    report_version,
    microns_per_pixel=0.496,
    invasive_margin_width_microns = 40,
    invasive_margin_drawn_line_width_pixels = 10,
    processes = 1,
    gimp_repositioned = False,
    verbose=False
    ):
    """
    Perform the functionality of ipiris

    Args:
        path (string): A file path to a directory containing the FOXP3 and PD1_PDL1 and (optionally GIMP) directories
        sample_name (string): The name of the sample
        panel_source (string): The file path to panel definitions
        panel_name (string): The name of the panel to use
        panel_version (string): The version of the panel to use
        report_source (string): The file path to report definitions
        report_name (string): The name of the report to use
        report_version (string): The version of the report to use
        microns_per_pixel (float): The number to convert between microns and pixels (default: 0.496 for Polaris 20x)
        invasive_margin_width_microns (float): Size in microns to expand the margin in one direction, radial width (default: 40 microns)
        invasive_margin_drawn_line_width_pixels (int): Width in pixels of the drawn line if present (default: 10 pixels)
        processes (int): The number of processes to use >1 will use multiprocessing to do multiple ROIs at once (default: 1)
        gimp_repositioned (bool): If True, the GIMP files are duplicated into the FOXP3 and PD1_PDL1 folders, if False, expect FOXP3, PD1_PDL1 and GIMP folders (default: False)
        verbose (bool): Output more run status (default: False)
    Returns:
        full_report (dict): The ip-iris style report
        csi (CellSampleInForm): Pythologist reader sample-level file
        dfs (dict of pandas dataframes): sample and ROI level reports for count and percentage data

    """
    panels = json.loads(open(panel_source,'r').read())
    reports = json.loads(open(report_source,'r').read())
    # read in the data
    csi = CellSampleInFormImmunoProfile()

    step_size = round((invasive_margin_width_microns/microns_per_pixel)-(invasive_margin_drawn_line_width_pixels/2))
    if verbose:
        sys.stderr.write("Step size for watershed: "+str(step_size)+"\n")
    csi.read_path(path=path,
              sample_name=sample_name,
              panel_name=panel_name,
              panel_version=panel_version,
              panels=panels,
              verbose=verbose,
              steps=step_size,
              processes=processes,
              skip_segmentation_processing=True,
              gimp_repositioned=gimp_repositioned
            )

    # make the cell dataframe
    cdf = csi.cdf(region_group='InFormLineArea',
              mutually_exclusive_phenotypes=panels[panel_name][panel_version]['phenotypes']).\
    drop_regions(['Undefined','OuterStroma'])#.filter_regions_by_area_pixels()
    cdf.microns_per_pixel = microns_per_pixel

    regions = {}
    # Drop everything except the region we are using in each one
    #included_margin = []
    #if 'InnerMargin' in cdf.regions:
    #    included_margin += ['InnerMargin']
    #if 'Margin' in cdf.regions:
    #    included_margin += ['InnerMargin']
    regions['Tumor + Invasive Margin'] = cdf.\
        combine_regions(['InnerMargin', 'InnerTumor', 'OuterMargin'],'Tumor + Invasive Margin')
    regions['Invasive Margin'] = cdf.\
        combine_regions(['InnerMargin', 'OuterMargin'],'Invasive Margin')
    regions['Tumor'] = cdf.\
        combine_regions(['InnerTumor'],'Tumor')
    regions['Full Tumor'] = cdf.\
        combine_regions(['InnerTumor','InnerMargin'],'Full Tumor')

    # Now read through and build the report
    sample_count_densities = []
    sample_count_percentages = []

    frame_count_densities = []
    frame_count_percentages = []

    # reconstruct the ugly R report
    report = []

    report_format = reports[report_name][report_version]
    for _row_number, _report_row in enumerate(report_format['report_rows']):
        orow = _report_row.copy()
        orow['row_number'] = _row_number + 1
        if 'phenotype' in orow:
            del orow['phenotype']
        del orow['preprocessing']
        if 'denominator_phenotypes' in orow:
            del orow['denominator_phenotypes']
        if 'numerator_phenotypes' in orow:
            del orow['numerator_phenotypes']
        #print(orow)

        _region_name = _report_row['region_label'] if 'region_label' in\
                   _report_row else _report_row['region']
        _cdf = regions[_region_name].copy()
        # do collapse preprocessing if we need to combine phenotypes
        for _collapse in _report_row['preprocessing']['collapse_phenotypes']:
            _cdf = _cdf.collapse_phenotypes(_collapse['inputs'],
                                        _collapse['output'])
        # do gating preprocessing
        for _gate in _report_row['preprocessing']['gates']:
            _cdf = _cdf.threshold(_gate['phenotype'],_gate['label'])

        # Get our counts object
        _measured_regions = _cdf.get_measured_regions().loc[_cdf.get_measured_regions()['region_label']==_region_name]
        _counts = _cdf.counts(measured_regions=_measured_regions)
    
        # now extract features
        if _report_row['test'] == 'Count Density':
            _pop1 = [SL(phenotypes=[_report_row['phenotype']],
                       label=_report_row['biomarker_label'])]
            _scnts = _counts.sample_counts(_pop1)
            _fcnts = _counts.frame_counts(_pop1)
            _scnts['row_number'] = orow['row_number']
            _fcnts['row_number'] = orow['row_number']
            _scnts['test'] = _report_row['test']
            _fcnts['test'] = _report_row['test']
            sample_count_densities.append(_scnts)
            frame_count_densities.append(_fcnts)
            _scnts = _scnts\
                [['sample_name',
                  'mean_density_mm2',
                  'stderr_density_mm2',
                  'stddev_density_mm2',
                  'frame_count',
                  'measured_count',
                ]].\
                rename(columns={'mean_density_mm2':'count_density',
                                'stderr_density_mm2':'standard_error',
                                'stddev_density_mm2':'standard_deviation',
                                'measured_count':'measured_count',
                                'sample_name':'sample'
                               })
            odata = _scnts.iloc[0].to_dict()
        if _report_row['test'] == 'Percent Population':
            _pop2 = [PL(numerator=SL(phenotypes=_report_row['numerator_phenotypes']),
                        denominator=SL(phenotypes=_report_row['denominator_phenotypes']),
                        label = _report_row['biomarker_label']
                       )
                    ]
            _spcts = _counts.sample_percentages(_pop2)
            _fpcts = _counts.frame_percentages(_pop2)
            _spcts['row_number'] = orow['row_number']
            _fpcts['row_number'] = orow['row_number']
            _spcts['test'] = _report_row['test']
            _fpcts['test'] = _report_row['test']
            sample_count_percentages.append(_spcts)
            frame_count_percentages.append(_fpcts)
            _spcts = _spcts\
            [['sample_name',
              'mean_percent',
              'stderr_percent',
              'stddev_percent',
              'cumulative_numerator',
              'cumulative_denominator',
              'cumulative_percent',
              'frame_count',
              'measured_count'
             ]].rename(columns={
            'cumulative_numerator':'cumulative_numerator_count',
            'cumulative_denominator':'cumulative_denominator_count',
            'measured_count':'measured_count',
            'stderr_percent':'standard_error_percent',
            'stddev_percent':'standard_deviation_percent',
            'qualified_frame_count':'measured_count',
            'sample_name':'sample'
            })
            _spcts['mean_fraction'] = _spcts['mean_percent'].\
                apply(lambda x: np.nan if x!=x else x/100)
            _spcts['standard_error_fraction'] = _spcts['standard_error_percent'].\
                apply(lambda x: np.nan if x!=x else x/100)
            _spcts['standard_deviation_fraction'] = _spcts['standard_deviation_percent'].\
                apply(lambda x: np.nan if x!=x else x/100)
            _spcts['cumulative_fraction'] = _spcts['cumulative_percent'].\
                apply(lambda x: np.nan if x!=x else x/100)
            odata = _spcts.iloc[0].to_dict()
        if _report_row['test'] == 'Population Area':
            _pop3 = [SL(phenotypes=[_report_row['phenotype']],
                        label=_report_row['biomarker_label'])]
            _sarea = _counts.sample_counts(_pop3)
            _farea = _counts.frame_counts(_pop3)
            _sarea['row_number'] = orow['row_number']
            _farea['row_number'] = orow['row_number']
            _sarea['test'] = _report_row['test']
            _farea['test'] = _report_row['test']
            sample_count_densities.append(_sarea)
            frame_count_densities.append(_farea)
            _sarea = _sarea\
                [['sample_name',
                  'cumulative_area_coverage_percent',
                  'cumulative_region_area_pixels',
                  'cumulative_cell_area_pixels',
                  'frame_count',
                  'mean_area_coverage_percent',
                  'stderr_area_coverage_percent',
                  'stddev_area_coverage_percent'
                 ]].rename(columns={
                'cumulative_cell_area_pixels':'area',
                'cumulative_region_area_pixels':'total_area',
                'cumulative_area_coverage_percent':'cumulative_coverage_percent',
                'mean_area_coverage_percent':'mean_coverage_percent',
                'stderr_area_coverage_percent':'stderr_coverage_percent',
                'stddev_area_coverage_percent':'stddev_coverage_percent',
                'sample_name':'sample'
                })
            _sarea['mean_coverage_fraction']=_sarea['mean_coverage_percent'].\
                apply(lambda x: np.nan if x!=x else x/100)
            _sarea['stderr_coverage_fraction']=_sarea['stderr_coverage_percent'].\
                apply(lambda x: np.nan if x!=x else x/100)
            _sarea['stddev_coverage_fraction']=_sarea['stddev_coverage_percent'].\
                apply(lambda x: np.nan if x!=x else x/100)
            _sarea['cumulative_coverage_fraction']=_sarea['cumulative_coverage_percent'].\
                apply(lambda x: np.nan if x!=x else x/100)
            odata = _sarea.iloc[0].to_dict()
            # if you're in this then change region
            orow['region_label'] = orow['region']
            del orow['region']
        output_set = {
            'row_layout':orow,
            'output':odata
        }
        report.append(output_set)
    sample_count_densities = pd.concat(sample_count_densities).reset_index(drop=True)
    sample_count_percentages = pd.concat(sample_count_percentages).reset_index(drop=True)

    frame_count_densities = pd.concat(frame_count_densities).reset_index(drop=True)
    frame_count_percentages = pd.concat(frame_count_percentages).reset_index(drop=True)

    full_report = {
        'sample':sample_name,
        'report':report,
        'QC':{},
        'meta':{
        
        },
        'parameters':{
            'inform_analysis_path':path,
            'invasive_margin_drawn_line_width_pixels':invasive_margin_drawn_line_width_pixels,
            'invasive_margin_width_microns':invasive_margin_width_microns,
            'microns_per_pixel':microns_per_pixel,
            'panel_name':panel_name,
            'panel_source':panel_source,
            'panel_version':panel_version,
            'report_name':report_name,
            'report_source':report_source,
            'report_version':report_version,
            'sample_name':sample_name
        }
    }

    # Get a regions dataframe
    _t1 = cdf.get_measured_regions()
    _t1 = _t1.loc[_t1['region_label']=='InnerTumor',:].rename(columns={'InnerTumor':'Tumor'})

    _t2 = cdf.get_measured_regions()
    _t2 = _t2.loc[_t2['region_label'].isin(['InnerTumor','InnerMargin','OuterMargin']),:]
    _t2['region_label'] = 'Tumor + Invasive Margin'
    _t2 = _t2.groupby(cdf.frame_columns+['region_label']).sum().reset_index()

    _t3 = cdf.get_measured_regions()
    _t3 = _t3.loc[_t3['region_label'].isin(['InnerMargin','OuterMargin']),:]
    _t3['region_label'] = 'Invasive Margin'
    _t3 = _t3.groupby(cdf.frame_columns+['region_label']).sum().reset_index()

    _t4 = cdf.get_measured_regions()
    _t4 = _t4.loc[_t4['region_label'].isin(['InnerTumor','InnerMargin']),:]
    _t4['region_label'] = 'Full Tumor'
    _t4 = _t4.groupby(cdf.frame_columns+['region_label']).sum().reset_index()

    regions = pd.concat([_t1,_t2,_t3,_t4]).\
        drop(columns=['project_id','project_name','sample_id','frame_id','region_cell_count'])
    regions['region_area_mm2'] = regions['region_area_pixels'].apply(lambda x: x*(cdf.microns_per_pixel*cdf.microns_per_pixel)/1000000)


    dfs = {
        'sample_count_densities':sample_count_densities.drop(columns=['project_id','project_name','sample_id','population_percent','sample_total_count']),
        'sample_count_percentages':sample_count_percentages.drop(columns=['project_id','project_name','sample_id']),
        'frame_count_densities':frame_count_densities.drop(columns=['project_id','project_name','sample_id','frame_id','population_percent','frame_total_count']),
        'frame_count_percentages':frame_count_percentages.drop(columns=['project_id','project_name','sample_id','frame_id']),
        'regions':regions
    }
    return full_report, csi, dfs


def report_dict_to_dataframes(report_dict):
    sample_name = report_dict['sample']
    count_densities = []
    count_percentages = []
    count_areas = []
    for _row in report_dict['report']:
        if _row['row_layout']['test'] == 'Count Density': 
            _row['output']['region'] = _row['row_layout']['region']
            _row['output']['biomarker_label'] = _row['row_layout']['biomarker_label']
            _row['output']['row_number'] = _row['row_layout']['row_number']
            _row['output']['test'] = _row['row_layout']['test']
            count_densities.append(pd.Series(_row['output']))
        if _row['row_layout']['test'] == 'Percent Population': 
            _row['output']['region'] = _row['row_layout']['region_label']
            _row['output']['biomarker_label'] = _row['row_layout']['biomarker_label']
            _row['output']['row_number'] = _row['row_layout']['row_number']
            _row['output']['test'] = _row['row_layout']['test']
            count_percentages.append(pd.Series(_row['output']))
        if _row['row_layout']['test'] == 'Population Area': 
            _row['output']['region'] = _row['row_layout']['region_label']
            _row['output']['biomarker_label'] = _row['row_layout']['biomarker_label']
            _row['output']['row_number'] = _row['row_layout']['row_number']
            _row['output']['test'] = _row['row_layout']['test']
            count_areas.append(pd.Series(_row['output']))
    count_densities = pd.DataFrame(count_densities)
    count_percentages = pd.DataFrame(count_percentages)
    count_areas = pd.DataFrame(count_areas)
    return count_densities.loc[:,sorted(count_densities.columns)],\
           count_percentages.loc[:,sorted(count_percentages.columns)],\
           count_areas.loc[:,sorted(count_areas.columns)]