""" Extract reports from the pipeline output data.

Input is an output_json prepared by the run tool.
"""

from importlib_resources import files
from pythologist_schemas import get_validator
import logging, argparse, json
import pandas as pd
from collections import OrderedDict

def cli():
    args = do_inputs()
    main(args)
def _prepend(d,input_key,input_value):
    d2 = [(k,v) for k,v in d.items()]
    d2 = [(input_key,input_value)] + d2
    return OrderedDict(d2)
def _append(d,input_key,input_value):
    d2 = [(k,v) for k,v in d.items()]
    d2 = d2+[(input_key,input_value)]
    return OrderedDict(d2)

def main(args):
    "We need to take the platform and return an appropriate input template"
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.WARNING)
    logger = logging.getLogger("report extraction")
    report = json.loads(open(args.report_json,'rt').read())
    logger.info("check report json format")
    get_validator(files('schema_data').joinpath('report_output.json')).\
        validate(report)
    logger.info("report json validated")

    # Lets start formulating the dataframes

    # cache these in arrays here but concatonate to dataframes the end
    sheets = OrderedDict({
        'smp_cnt_cumulative_lf':[],
        'smp_cnt_aggregate_lf':[],
        'smp_pct_cumulative_lf':[],
        'smp_pct_aggregate_lf':[],
        'img_cnt_lf':[],
        'img_pct_lf':[]
    })
    info = {}
    for sample in report['sample_outputs']:
        sample_name = sample['sample_name']
        _df = pd.DataFrame([row for row in sample['sample_reports']['sample_cumulative_count_densities']])
        _df['sample_name'] = sample_name
        sheets['smp_cnt_cumulative_lf'].append(_df)
        info['smp_cnt_cumulative_lf'] = {
            'index':False,
            'description':'sample-level count density measurements treating all ROIs as a single large image in long table format.'
        }

        _df = pd.DataFrame([row for row in sample['sample_reports']['sample_aggregate_count_densities']])
        _df['sample_name'] = sample_name
        sheets['smp_cnt_aggregate_lf'].append(_df)
        info['smp_cnt_aggregate_lf'] = {
            'index':False,
            'description':'sample-level count density measurements averaging the measures from ROIs in long table format.'
        }

        _df = pd.DataFrame([row for row in sample['sample_reports']['sample_cumulative_count_percentages']])
        _df['sample_name'] = sample_name
        sheets['smp_pct_cumulative_lf'].append(_df)
        info['smp_pct_cumulative_lf'] = {
            'index':False,
            'description':'sample-level percentage measurements treating all ROIs as a single large image in long table format.'
        }

        _df = pd.DataFrame([row for row in sample['sample_reports']['sample_aggregate_count_percentages']])
        _df['sample_name'] = sample_name
        sheets['smp_pct_aggregate_lf'].append(_df)
        info['smp_pct_aggregate_lf'] = {
            'index':False,
            'description':'sample-level percentage measurements averaging the measures from ROIs in long table format.'
        }

        # Now get the images
        for image in sample['images']:
            image_name = image['image_name']
            _df = pd.DataFrame([row for row in image['image_reports']['image_count_densities']])
            _df['sample_name'] = sample_name
            _df['image_name'] = image_name
            sheets['img_cnt_lf'].append(_df)
            info['img_cnt_lf'] = {
                'index':False,
                'description':'image-level count density measurements in long table format.'
            }

            _df = pd.DataFrame([row for row in image['image_reports']['image_count_percentages']])
            _df['sample_name'] = sample_name
            _df['image_name'] = image_name
            sheets['img_pct_lf'].append(_df)
            info['img_pct_lf'] = {
                'index':False,
                'description':'image-level percentage measurements in long table format.'
            }

    # move the sheets to dataframes
    for sheet_name in sheets:
        sheets[sheet_name] = pd.concat(sheets[sheet_name])


    _nums = sheets['img_pct_lf'][['sample_name','image_name']].drop_duplicates().sort_values('image_name').reset_index(drop=True).\
        groupby('sample_name').apply(lambda x: pd.Series(dict(zip(
            x['image_name'],
            [y+1 for y in range(0,len(x['image_name']))]
        )))).reset_index().rename(columns={'level_1':'image_name',0:'idx'})
    _df = _nums.merge(sheets['img_pct_lf'],on=['sample_name','image_name']).set_index(['region_name','population_name','sample_name'])[['idx','percent']].\
        pivot(columns='idx')
    _df.columns = [str(x) for x in _df.columns.droplevel(0)]
    sheets = _prepend(sheets,'img_pct_mat',_df)
    info['img_pct_mat'] = {
                'index':True,
                'description':'image-level percentage measurement in matrix format.'
            }

    _nums = sheets['img_cnt_lf'][['sample_name','image_name']].drop_duplicates().sort_values('image_name').reset_index(drop=True).\
        groupby('sample_name').apply(lambda x: pd.Series(dict(zip(
            x['image_name'],
            [y+1 for y in range(0,len(x['image_name']))]
        )))).reset_index().rename(columns={'level_1':'image_name',0:'idx'})
    _df = _nums.merge(sheets['img_cnt_lf'],on=['sample_name','image_name']).set_index(['region_name','population_name','sample_name'])[['idx','density_mm2']].\
        pivot(columns='idx')
    _df.columns = [str(x) for x in _df.columns.droplevel(0)]
    sheets = _prepend(sheets,'img_cnt_mat',_df)
    info['img_cnt_mat'] = {
                'index':True,
                'description':'image-level count density measurement in matrix format.'
            }

    # Add some classic style reports
    _df = sheets['smp_pct_cumulative_lf'].\
        set_index(['region_name','sample_name','image_count'])[['cumulative_percent','population_name']].\
        pivot(columns='population_name')
    sheets = _prepend(sheets,'smp_pct_cumulative_mat',_df)
    info['smp_pct_cumulative_mat'] = {
                'index':True,
                'description':'sample-level percentage measurements treating all ROIs as a single large image in matrix format.'
            }

    # Add some classic style reports
    _df = sheets['smp_pct_aggregate_lf'].rename(columns={'aggregate_mean_percent':'mean_percent','aggregate_stderr_percent':'stderr_percent'}).\
        set_index(['region_name','sample_name','image_count'])[['mean_percent','stderr_percent','population_name']].\
        pivot(columns='population_name')
    _df = _df.swaplevel(axis=1).sort_index(1)
    sheets = _prepend(sheets,'smp_pct_aggregate_mat',_df)
    info['smp_pct_aggregate_mat'] = {
                'index':True,
                'description':'sample-level percentage measurements averaging the measures from ROIs in matrix format.'
            }

    # Add some classic style reports
    _df = sheets['smp_cnt_cumulative_lf'].\
        set_index(['region_name','sample_name','image_count'])[['cumulative_density_mm2','population_name']].\
        pivot(columns='population_name')
    sheets = _prepend(sheets,'smp_cnt_cumulative_mat',_df)
    info['smp_cnt_cumulative_mat'] = {
                'index':True,
                'description':'sample-level count density measurements treating all ROIs as a single large image in matrix format.'
            }

    # Add some classic style reports
    _df = sheets['smp_cnt_aggregate_lf'].rename(columns={'aggregate_mean_density_mm2':'mean_density_mm2','aggregate_stderr_density_mm2':'stderr_density_mm2'}).\
        set_index(['region_name','sample_name','image_count'])[['mean_density_mm2','stderr_density_mm2','population_name']].\
        pivot(columns='population_name')
    _df = _df.swaplevel(axis=1).sort_index(1)
    sheets = _prepend(sheets,'smp_cnt_aggregate_mat',_df)
    info['smp_cnt_aggregate_mat'] = {
                'index':True,
                'description':'sample-level count density measurements averaging the measures from ROIs in matrix format.'
            }

    writer = pd.ExcelWriter(args.output_excel, engine='xlsxwriter')

    for sheetname, df in sheets.items():  # loop through `dict` of dataframes
        df.to_excel(writer, sheet_name=sheetname,float_format="%.2f",index=info[sheetname]['index'])  # send df to writer
        worksheet = writer.sheets[sheetname]  # pull worksheet object
        for idx, clen in enumerate([20]*(df.index.nlevels if info[sheetname]['index'] else 0)+\
                                   [len(x)+1 if isinstance(x,str) else max([len(y) for y in x])+1 for x in df.columns]):  # loop through all columns
            max_len = max(8,clen*1.1)
            worksheet.set_column(idx, idx, max_len)  # set column width

    writer.save()

    return 

def do_inputs():
    parser = argparse.ArgumentParser(
            description = "Run the pipeline",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--report_json',required=True,help="The report json that was output but the run")
    parser.add_argument('--output_excel',required=True,help="The path to write the output excel report")
    parser.add_argument('--verbose',action='store_true',help="Show more about the run")
    args = parser.parse_args()
    return args

def external_cmd(cmd):
    """function for calling program by command through a function"""
    cache_argv = sys.argv
    sys.argv = cmd
    args = do_inputs()
    main(args)
    sys.argv = cache_argv


if __name__ == "__main__":
    main(do_inputs())