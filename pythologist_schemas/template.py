from openpyxl import Workbook, load_workbook
#from importlib_resources import files
from pythologist_schemas import get_validator

def excel_to_json(excel_template_path,
                  json_schema_path,
                  sheet_names,
                  parameter_sheet="Parameters",
                  #do_parameters=True,
                  ignore_extra_parameters=True):
    """
    Read the analysis data from a filled-in template file
    into a json object compatible with its respective json-schema.
    Return back the json object or None, whether its valid or not, and any errors.
    """
    #_fname = files('schema_data.inputs.platforms.InForm').joinpath('analysis.json')
    _validator = get_validator(json_schema_path)
    wb = load_workbook(excel_template_path)
    
    # Create the object we will save the data in
    output = {
        
    }
    
    # Lets do the Parameters first    
    _parameter_key, _parameters, parameters_success, parameters_errors = _read_parameters("Parameters",
                                                   wb,
                                                   _validator.schema,
                                                   ignore_extra_parameters=ignore_extra_parameters)
    output[_parameter_key] = _parameters

    #sheets = ['Exports','Mutually Exclusive Phenotypes','Binary Phenotypes','Regions']
    total_success = True
    total_errors = []
    for sheet in sheet_names:
        #print(sheet)
        # Lets do the Repeating fields next
        _repeating_key, _data, repeat_success, repeat_errors = _read_repeating(sheet,wb,_validator.schema)
        total_success = total_success and repeat_success
        total_errors += repeat_errors
        output[_repeating_key] = _data
    
    pass_validation = True
    try:
        _validation = _validator.validate(instance=output)
    except:
        pass_validation = False
        raise

        
    analysis_success = total_success and pass_validation
    if not analysis_success: output = None
    return output, \
           analysis_success, \
           total_errors

def _read_parameters(worksheet_title,workbook,schema,ignore_extra_parameters=False):
    """
    Return the property (string), the data (object), and whether this step succeded (bool), and any errors
    """
    _keyname = None
    for x in schema['properties']:
        if 'title' in schema['properties'][x] and \
            schema['properties'][x]['title']==worksheet_title:
            _keyname = x
            break
    if _keyname is None:
        raise ValueError('Unable to find a property with the title "'+str(worksheet_title)+'" while reading parameters')

    # get our expected parameter list
    ws = workbook[worksheet_title]
    _dict = dict([[y.value for y in x] for x in ws][1:])
    _trans = dict([(schema['properties'][_keyname]['properties'][x]['title'],x) 
                   for x in schema['properties'][x]['properties'].keys()])
    if ignore_extra_parameters:
        for _k in list(_dict.keys()):
            if _k not in _trans.keys():
                del _dict[_k]
    # for whats left add the ekeys
    _parameters = {}
    for _k in _dict:
        _parameters[_trans[_k]] = _dict[_k]
    return _keyname, _parameters, True, []

def _read_repeating(worksheet_title,workbook,schema):
    """
    Return the property (string), the data (object), and whether this step succeded (bool), and any errors
    """
    _keyname = None
    for x in schema['properties']:
        if 'title' in schema['properties'][x] and \
            schema['properties'][x]['title']==worksheet_title:
            _keyname = x
            break
    if _keyname is None:
        raise ValueError('Unable to find a property with the title "'+str(worksheet_title)+'" while reading repeating data')

    #print(_keyname)
    # get our expected parameter list
    ws = workbook[worksheet_title]

    # Start by reading in the header and its conversion to propertys
    _header = [y.value for y in [x for x in ws][0]]
    #print(_header)
    _trans = dict([(schema['properties'][_keyname]['items']['properties'][x]['title'],x) \
                   for x in schema['properties'][_keyname]['items']['properties']])
    
    for column_name in _header:
        if column_name not in _trans:
            raise ValueError('Column Name "'+str(column_name)+'" is not defined among the titles in the json-schema for data table "'+str(_keyname)+'""')
    
    _data = {}
    if len([x for x in ws]) <= 1: 
        _data = []
    else:
        _data = [dict(zip([_trans[z] for z in _header],[y.value for y in x])) for x in ws][1:]

    return _keyname, _data, True, []

