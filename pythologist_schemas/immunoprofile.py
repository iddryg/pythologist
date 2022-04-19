from pythologist_reader.formats.inform.immunoprofile import CellSampleInFormImmunoProfile
from pythologist import CellDataFrame, SubsetLogic as SL, PercentageLogic as PL
import pandas as pd
import json, sys

def execute_immunoprofile_extraction(
    path,
    sample_name,
    panel_source,
    panel_name,
    report_source,
    panel_version,
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
    drop_regions(['Undefined','OuterStroma']).filter_regions_by_area_pixels()
    cdf.microns_per_pixel = microns_per_pixel

    regions = {}
    # Drop everything except the region we are using in each one
    regions['Tumor + Invasive Margin'] = cdf.\
        combine_regions(['InnerMargin', 'InnerTumor', 'OuterMargin'],'Tumor + Invasive Margin')
    regions['Invasive Margin'] = cdf.\
        combine_regions(['InnerMargin', 'OuterMargin'],'Invasive Margin').\
        drop_regions(['InnerTumor'])
    regions['Tumor'] = cdf.\
        drop_regions(['InnerMargin', 'OuterMargin']).\
        combine_regions(['InnerTumor'],'Tumor')
    regions['Full Tumor'] = cdf.\
        combine_regions(['InnerTumor','InnerMargin'],'Full Tumor').\
        drop_regions(['OuterMargin'])

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
        _counts = _cdf.counts()
    
        # now extract features
        if _report_row['test'] == 'Count Density':
            _pop1 = [SL(phenotypes=[_report_row['phenotype']],
                       label=_report_row['biomarker_label'])]
            _scnts = _counts.sample_counts(_pop1)
            _fcnts = _counts.frame_counts(_pop1)
            _scnts['row_number'] = orow['row_number']
            _fcnts['row_number'] = orow['row_number']
            sample_count_densities.append(_scnts)
            frame_count_densities.append(_fcnts)
            _scnts = _scnts\
                [['sample_name',
                  'region_label',
                  'phenotype_label',
                  'mean_density_mm2',
                  'stderr_density_mm2',
                  'stddev_density_mm2',
                  'frame_count',
                  'measured_frame_count',
                ]].\
                rename(columns={'mean_density_mm2':'count_density',
                                'stderr_density_mm2':'standard_error',
                                'stddev_density_mm2':'standard_deviation',
                                'measured_frame_count':'measured_count'
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
            sample_count_percentages.append(_spcts)
            frame_count_percentages.append(_fpcts)
            _spcts = _spcts\
            [['sample_name',
              'region_label',
              'phenotype_label',
              'mean_percent',
              'stderr_percent',
              'stddev_percent',
              'cumulative_numerator',
              'cumulative_denominator',
              'cumulative_percent',
              'frame_count',
              'qualified_frame_count'
             ]].rename(columns={
            'cumulative_numerator':'cumulative_numerator_count',
            'cumulative_denominator':'cumulative_denominator_count',
            'measured_frame_count':'measured_count',
            'stderr_percent':'standard_error_percent',
            'stddev_percent':'standard_deviation_percent',
            'qualified_frame_count':'measured_count'
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
            _sarea['row_number'] = orow['row_number']
            _sarea = _sarea\
                [['sample_name',
                  'region_label',
                  'phenotype_label',
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
                'stddev_area_coverage_percent':'stddev_coverage_percent'
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
    dfs = {
        'sample_count_densities':sample_count_densities.drop(columns=['project_id','project_name','sample_id','population_percent','sample_total_count']),
        'sample_count_percentages':sample_count_percentages.drop(columns=['project_id','project_name','sample_id']),
        'frame_count_densities':frame_count_densities.drop(columns=['project_id','project_name','sample_id','frame_id','population_percent','frame_total_count']),
        'frame_count_percentages':frame_count_percentages.drop(columns=['project_id','project_name','sample_id','frame_id'])
    }
    return full_report, csi, dfs