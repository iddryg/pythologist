"""
The CLI tool for validating the setup of templates, and the staging of files.  

"""
import argparse, os, json, sys, hashlib
from importlib_resources import files
from pythologist_schemas.template import excel_to_json
from pythologist_schemas.platforms.InForm.files import injest_project, injest_sample
from pythologist_schemas.report import convert_report_definition_to_report
from pythologist_reader.formats.inform.custom import CellFrameInFormLineArea, CellFrameInFormCustomMask
from pythologist_reader.formats.inform.frame import CellFrameInForm
from pythologist_image_utilities import hash_tiff_contents
import pandas as pd
import logging

sys.setrecursionlimit(15000)

def main(args):
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG,filename=args.output_log)
    else:
        logging.basicConfig(level=logging.WARNING,filename=args.output_log)
    logger = logging.getLogger("main")

    #make sure we can do the outputs if they are set
    if args.output_json and not args.report_excel:
        raise ValueError("cannot output a run setup without a report_excel")

    total_success = True

    logger.info("checking panel from analysis excel")

    _fname = files('schema_data.inputs').joinpath('panel.json')
    panel_json, panel_success, panel_errors  = excel_to_json(args.analysis_excel, \
                                                              _fname,
                                                              ['Panel'], \
                                                              ignore_extra_parameters=True)

    total_success = total_success and panel_success
   
    logger.info("checking the rest of the analysis excel")

    _fname = files('schema_data.inputs.platforms.InForm').joinpath('analysis.json')
    analysis_json, analysis_success, analysis_errors  = excel_to_json(args.analysis_excel, \
                                                                       _fname,
                                                                       ['Exports','Mutually Exclusive Phenotypes','Binary Phenotypes','Regions'], \
                                                                       ignore_extra_parameters=True)

    # Check to make sure we have requirements specific to special cases
    if analysis_json['parameters']['region_annotation_strategy'] == 'GIMP_TSI' and \
       (analysis_json['parameters']['draw_margin_width'] is None or analysis_json['parameters']['expanded_margin_width_um'] is None):
       raise ValueError("When region strategy is GIMP_TSI, analysis Draw Margin Width and Expanded Margin Width must be set")
    if analysis_json['parameters']['region_annotation_strategy'] == 'GIMP_CUSTOM' and \
       (analysis_json['parameters']['region_annotation_custom_label'] is None or analysis_json['parameters']['unannotated_region_label'] is None):
       raise ValueError("When region strategy is GIMP_CUSTOM, analysis Region Anntoation Custom Label and Unannotated Region Label must be set")

    total_success = total_success and analysis_success

    logger.info("checking the project excel")

    _fname = files('schema_data.inputs.platforms.InForm').joinpath('project.json')
    project_json, project_success, project_errors  = excel_to_json(args.project_excel, \
                                                                   _fname,
                                                                   ['Samples'], \
                                                                   ignore_extra_parameters=False)

    total_success = total_success and project_success

    project_path, _tmp = os.path.split(os.path.abspath(args.project_excel))



    total_mutually_exclusive_phenotypes, total_binary_phenotype_target_names = _allowed_phenotypes(analysis_json)

    ## 1b. Read in the 'not absolutely necessary for end-to-end run' report


    if args.report_excel:
        logger.info("checking the report excel")
        _fname = files('schema_data.inputs').joinpath('report_definition.json')
        report_definition_json, report_definition_success, report_definition_errors  = excel_to_json(args.report_excel, \
                                                                    _fname,
                                                                    ['Region Selection','Population Percentages','Population Densities'], \
                                                                    ignore_extra_parameters=False)
        total_success = total_success and report_definition_success

        report_json = convert_report_definition_to_report(report_definition_json)

        report_compatibility_success, report_compatibility_errors = _report_compatibility(report_json,
                                                                          total_mutually_exclusive_phenotypes,
                                                                          total_binary_phenotype_target_names,
                                                                          [x['region_name'] for x in analysis_json['regions']])

        total_success = total_success and report_compatibility_success

    # 2. No we can ensure the files are properly structured

    if args.sample_name: 
        logger.info("checking the structure of specific sample "+str(args.sample_name))
        sample_file, injestion_success, injest_errors = injest_sample(args.sample_name,project_json,analysis_json,project_path)   
        sample_files = [sample_file]
    else:
        logger.info("checking entire project")
        sample_files, injestion_success, injest_errors = injest_project(project_json,analysis_json,project_path)
    total_success = total_success and injestion_success

    # 3. Now we can run pythologist to get a light read on each sample.

    for sample_file in sample_files:
        logger.info("checking sample "+str(sample_file['sample_name']))
        _lightly_validate_sample(sample_file,analysis_json,project_json,panel_json,project_path)

    if total_success:
        logger.info("All tests passed.")

    if not args.report_excel or not total_success:
        return None

    output = {
        'project':project_json,
        'panel':panel_json,
        'analysis':analysis_json,
        'report':report_json,
        'sample_files':sample_files
    }
    if args.output_json:
        with open(args.output_json,'wt') as of:
            of.write(json.dumps(output,indent=2))
    #print(json.dumps(output,indent=2))
    return 

def _allowed_phenotypes(analysis_json):
   # Extract the mutually exclusive phenotypes and binary phenotype target names
   _exports = [x['export_name'] for x in analysis_json['inform_exports'] if x['primary_phenotyping']]
   if len(_exports) == 0: raise ValueError("need at least one primary phenotyping")
   if len(_exports) > 1: raise ValueError("can only have at most one primary phenotyping")
   primary_export_name = _exports[0]

   # now get the phenotypes we expect on that export
   total_mutually_exclusive_phenotypes = [x['phenotype_name'] for x in analysis_json['mutually_exclusive_phenotypes'] if x['export_name']==primary_export_name]
   if len(total_mutually_exclusive_phenotypes) == 0: raise ValueError("Expecting phenotypes to be defined")

   # now get the binary phenotypes that we expect from any export
   total_binary_phenotype_target_names = [x['target_name'] for x in analysis_json['binary_phenotypes']]+\
                                         [x['phenotype_name'] for x in analysis_json['mutually_exclusive_phenotypes'] if x['convert_to_binary']]
   return total_mutually_exclusive_phenotypes, total_binary_phenotype_target_names

def _report_compatibility(report_json,total_mutually_exclusive_phenotypes,total_binary_phenotype_target_names,region_names):
    # start with the population densities
    logger = logging.getLogger("report contents")
    for i,measure in enumerate(report_json['population_densities']):
        logger.info("checking ability to measure population density for: "+str(measure['population_name']))
        _unknown = set(measure['mutually_exclusive_phenotypes'])-set(total_mutually_exclusive_phenotypes)
        if len(_unknown) > 0: raise ValueError("Density report contains a mutually exclusive phenotype(s) thats not accounted for "+str(_unknown))
        _unknown = set([x['target_name'] for x in measure['binary_phenotypes']])-set(total_binary_phenotype_target_names)
        if len(_unknown) > 0: raise ValueError("Density report contains a binary phenotype(s) thats not accounted for "+str(_unknown))
    for i,measure in enumerate(report_json['population_percentages']):
        logger.info("checking ability to measure population percentages for: "+str(measure['population_name']))
        _unknown = set(measure['numerator_mutually_exclusive_phenotypes'])-set(total_mutually_exclusive_phenotypes)
        if len(_unknown) > 0: raise ValueError("Percentage report contains a numerator mutually exclusive phenotype(s) thats not accounted for "+str(_unknown))
        _unknown = set([x['target_name'] for x in measure['numerator_binary_phenotypes']])-set(total_binary_phenotype_target_names)
        if len(_unknown) > 0: raise ValueError("Percentage report contains a numerator binary phenotype(s) thats not accounted for "+str(_unknown))
        _unknown = set(measure['denominator_mutually_exclusive_phenotypes'])-set(total_mutually_exclusive_phenotypes)
        if len(_unknown) > 0: raise ValueError("Percentage report contains a denominator mutually exclusive phenotype(s) thats not accounted for "+str(_unknown))
        _unknown = set([x['target_name'] for x in measure['denominator_binary_phenotypes']])-set(total_binary_phenotype_target_names)
        if len(_unknown) > 0: raise ValueError("Percentage report contains a denominator phenotype(s) thats not accounted for "+str(_unknown))
    logger.info("checking the uniqueness of report region names")
    report_region_names = [x['report_region_name'] for x in report_json['region_selection']]
    if len(set(report_region_names)) < len(report_region_names):
        raise ValueError("Error there is duplicates among the report region name")
    for i, measure in enumerate(report_json['region_selection']):
        _unknown = set(measure['regions_to_combine'])-set(region_names)
        if len(_unknown) > 0: raise ValueError("Region name to combine is not among defined regions "+str(_unknown))

    return True, []
def _lightly_validate_sample(sample_file,analysis_json,project_json,panel_json,project_directory):
    logger = logging.getLogger(str(sample_file['sample_name']))

    sample_name = sample_file['sample_name']
    to_compare = {}
    for export in sample_file['exports']:
      to_compare[export['export_name']] = _lightly_validate_export(sample_name,export,analysis_json,project_json,panel_json,project_directory)

    # Now for each comparison we must traverse each image to make sure the exports have the proper concordance
    tests = {}
    for export_name in to_compare:
      for image_name in to_compare[export_name]:
         logger.info("concordance "+str(export_name)+"|"+str(image_name))
         for test in to_compare[export_name][image_name]:
            if test not in tests:
               tests[test] = {}
            if image_name not in tests[test]:
               tests[test][image_name] = set()
            tests[test][image_name].add(to_compare[export_name][image_name][test])
            if len(tests[test][image_name]) > 1:
               raise ValueError("Discordant exports for image "+str(image_name)+" for "+str(test))

def _lightly_validate_export(sample_name,export,analysis_json,project_json,panel_json,project_directory):
   export_name = export['export_name']
   logger = logging.getLogger(str(export_name))
   export_path = os.path.join(project_directory,sample_name,export_name)
   to_compare = {}
   for image_frame in export['images']:
      to_compare[image_frame['image_name']] = _lightly_validate_image_frame(image_frame,export_name,analysis_json,panel_json)

   return to_compare

def _lightly_validate_image_frame(image_frame,export_name,analysis_json,panel_json):

   logger = logging.getLogger("deep image "+str(export_name)+"|"+str(image_frame['image_name']))
   # Get the conversions for the channel names
   _markers = panel_json['markers']
   _markers = dict([(x['full_name'],x['marker_name']) for x in _markers])

   logger.info("checking for microns")
   # Check for each Panel "Full Platform Name"
   with open(image_frame['image_data']['cell_seg_data_txt']['file_path'],'rt') as inf:
      firstline = inf.readline()
      names = [x['full_name'] for x in panel_json['markers']]
      for name in names:
          if name not in firstline:
            raise ValueError(image_frame['image_data']['cell_seg_data_txt']['file_path']+"\n"+'the panel "Full Platform Name":"'+str(name)+'" was not seen in the InForm analysis files check that the analysis template and the inform project have the naming you want.')


   logger.info("checking for microns")
   # Check for the microns issue
   with open(image_frame['image_data']['cell_seg_data_txt']['file_path'],'rt') as inf:
      firstline = inf.readline()
      if 'microns' in firstline:
         raise ValueError('Detected microns instead of pixels in cell seg data')

   logger.info("checking for membrane segmentation")
   # Check for the presence of membrane segmentation
   with open(image_frame['image_data']['cell_seg_data_txt']['file_path'],'rt') as inf:
      firstline = inf.readline()
      if not 'Entire' in firstline:
         raise ValueError('Failed to detected membrane-based segmentation in cell seg data')

   image_name = image_frame['image_name']

   # find out if we actually have an implied region annotation strategy of custom where we are missing the margin
   implied_region_annotation_strategy = analysis_json['parameters']['region_annotation_strategy']
   implied_region_annotation_custom_label = analysis_json['parameters']['region_annotation_custom_label']
   implied_unannotated_region_label = analysis_json['parameters']['unannotated_region_label']
   # deal with the special case of a TSI missing a margin line. Treat it like a custom with only tumor
   if implied_region_annotation_strategy=='GIMP_TSI' and \
      len([x for x in image_frame['image_annotations'] if x['mask_label']=='TSI Line']) == 0:
      implied_region_annotation_strategy = 'GIMP_CUSTOM'
      implied_region_annotation_custom_label = 'Tumor'
      implied_unannotated_region_label = 'Stroma'

   if implied_region_annotation_strategy == 'GIMP_TSI':
      logger.info("reading GIMP_TSI format")
      cfi = CellFrameInFormLineArea()
      cfi.read_raw(frame_name = image_name,
                   cell_seg_data_file = image_frame['image_data']['cell_seg_data_txt']['file_path'],
                   score_data_file = None if 'score_data_txt' not in image_frame['image_data'] else \
                                     image_frame['image_data']['score_data_txt']['file_path'],
                   tissue_seg_data_file = None,
                   binary_seg_image_file = image_frame['image_data']['binary_segs_maps_tif']['file_path'],
                   component_image_file = image_frame['image_data']['component_data_tif']['file_path'],
                   verbose=False,
                   channel_abbreviations=_markers,
                   require=True,
                   require_score=False,
                   skip_segmentation_processing=False)

      line_image = [x for x in image_frame['image_annotations'] if x['mask_label']=='TSI Line'][0]
      tumor_image = [x for x in image_frame['image_annotations'] if x['mask_label']=='Tumor'][0]
      cfi.set_line_area(line_image['file_path'],
         tumor_image['file_path'],
         steps=analysis_json['parameters']['draw_margin_width'],
         verbose=True
         )
      cdf = cfi.cdf

   elif implied_region_annotation_strategy == 'GIMP_CUSTOM':
      logger.info("reading GIMP_CUSTOM format")
      cfi = CellFrameInFormCustomMask()
      cfi.read_raw(frame_name = image_name,
                   cell_seg_data_file = image_frame['image_data']['cell_seg_data_txt']['file_path'],
                   score_data_file = None if 'score_data_txt' not in image_frame['image_data'] else \
                                     image_frame['image_data']['score_data_txt']['file_path'],
                   tissue_seg_data_file = None,
                   binary_seg_image_file = image_frame['image_data']['binary_segs_maps_tif']['file_path'],
                   component_image_file = image_frame['image_data']['component_data_tif']['file_path'],
                   verbose=False,
                   channel_abbreviations=_markers,
                   require=True,
                   require_score=False,
                   skip_segmentation_processing=False)

      custom_image = [x for x in image_frame['image_annotations'] if x['mask_label']==implied_region_annotation_custom_label][0]
      cfi.set_area(custom_image['file_path'],
         implied_region_annotation_custom_label,
         implied_unannotated_region_label,
         verbose=True
         )
      cdf = cfi.cdf
   elif implied_region_annotation_strategy in ['NO_ANNOTATION','INFORM_ANALYSIS']:
      logger.info("reading standard InForm format")
      cfi = CellFrameInForm()
      cfi.read_raw(frame_name = image_name,
                   cell_seg_data_file = image_frame['image_data']['cell_seg_data_txt']['file_path'],
                   score_data_file = None if 'score_data_txt' not in image_frame['image_data'] else \
                                     image_frame['image_data']['score_data_txt']['file_path'],
                   tissue_seg_data_file = None,
                   binary_seg_image_file = image_frame['image_data']['binary_segs_maps_tif']['file_path'],
                   component_image_file = image_frame['image_data']['component_data_tif']['file_path'],
                   verbose=False,
                   channel_abbreviations=_markers,
                   require=True,
                   require_score=False,
                   skip_segmentation_processing=False)

      cdf = cfi.cdf
   else:
      raise ValueError("unknown annotation. you shouldn't see this with approrpiate enums")
   phenotypes = cdf.phenotypes
   binary_names = cdf.scored_names
   regions = cdf.regions

   # check phenotypes
   logger.info("check mutually exclusive phenotypes")
   expected_phenotypes = [x['phenotype_name'] for x in analysis_json['mutually_exclusive_phenotypes'] if x['export_name']==export_name]
   unexpected = list(set(phenotypes) - set(expected_phenotypes))
   if len(unexpected) > 0:
      raise ValueError("Image "+str(image_name)+" in "+str(export_name)+" contained unexpected phenotype(s) not defined in the analysis "+str(unexpected))
   missing = list(set(expected_phenotypes) - set(phenotypes))
   if len(missing) > 1:
      logger.warning("missing phenotype(s) "+str(missing))

   # check binary_names
   logger.info("check binary phenotypes")
   expected_binary_names = [x['target_name'] for x in analysis_json['binary_phenotypes'] if x['export_name']==export_name]
   unexpected = list(set(binary_names)-set(expected_binary_names))
   if len(unexpected) > 0:
      raise ValueError("Image "+str(image_name)+" in "+str(export_name)+" contained unexpected binary threshold target(s) not defined in the analysis "+str(unexpected))

   # Check regions
   logger.info("check regions")
   expected_regions = [x['region_name'] for x in analysis_json['regions'] ]
   # Write in an exception for if there are no regions and no annotations are given.
   if len(expected_regions)==0 and \
      analysis_json['parameters']['region_annotation_strategy'] == 'NO_ANNOTATION':
      raise ValueError("To use NO_ANNOTATION, you must define one region as 'Any'")
   if len(expected_regions)==1 and analysis_json['parameters']['region_annotation_strategy'] == 'NO_ANNOTATION' and \
      analysis_json['regions'][0]['region_name'] != 'Any':
      raise ValueError("If using NO_ANNOTATION you must define the only region to be 'Any'")

   unexpected = list(set(regions)-set(expected_regions))
   if len(unexpected) > 0:
      raise ValueError("Image "+str(image_name)+" in "+str(export_name)+" contained unexpected region(s) "+str(unexpected))
   
   logger.info("generate hashes of segmentation")
   seg_data = pd.read_csv(image_frame['image_data']['cell_seg_data_txt']['file_path'],sep="\t").\
       loc[:,['Cell ID', 'Cell X Position', 'Cell Y Position']].sort_values('Cell ID').\
       apply(lambda x: tuple(x),1).tolist()
   to_compare = {
      'binary seg file difference':hash_tiff_contents(image_frame['image_data']['binary_segs_maps_tif']['file_path']),
      'cell seg data segmentation difference':hashlib.sha256(json.dumps(tuple(seg_data)).encode('utf-8')).hexdigest()
   }
   return to_compare

def do_inputs():
   parser = argparse.ArgumentParser(
            description = "",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
   parser.add_argument('--project_excel',metavar='ProjectExcelPath',required=True,help="The path to a excel file of a filled-in project template.")
   parser.add_argument('--analysis_excel',metavar='AnalysisExcelPath',required=True,help="The path to an excel file of a filled-in analysis template.")
   parser.add_argument('--report_excel',metavar='ReportExcelPath',help="The path to an excel file of a filled-in report template.")
   parser.add_argument('--sample_name',metavar='SampleName',help="Only check one sample.")
   parser.add_argument('--output_log',help="Save the validation log")
   parser.add_argument('--output_json',help="Save the json that defines the run")
   parser.add_argument('--temp',help="Specify a temporary directory")
   parser.add_argument('--verbose',action='store_true',help="Report info and debug")
   args = parser.parse_args()
   return args

def cli():
   args = do_inputs()
   main(args)

def external_cmd(cmd):
   """function for calling program by command through a function"""
   cache_argv = sys.argv
   sys.argv = cmd
   args = do_inputs()
   main(args)
   sys.argv = cache_argv


if __name__ == "__main__":
   main(do_inputs())
