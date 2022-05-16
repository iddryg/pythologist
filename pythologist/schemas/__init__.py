import argparse, json, os, sys
from jsonschema import Draft7Validator, RefResolver, SchemaError

def get_validator(filename, base_uri=''):
	# Adapated from https://www.programcreek.com/python/example/83374/jsonschema.RefResolver
	# referencing code from HumanCellAtlas Apache License

    """Load schema from JSON file;
    Check whether it's a valid schema;
    Return a Draft4Validator object.
    Optionally specify a base URI for relative path
    resolution of JSON pointers (This is especially useful
    for local resolution via base_uri of form file://{some_path}/)
    """
    def get_json_from_file(filename):
        output = ''
        with open(filename,'rt') as f:
            output = f.read()
        return json.loads(output)
    schema = get_json_from_file(filename)
    try:
        # Check schema via class method call. Works, despite IDE complaining
        Draft7Validator.check_schema(schema)
        #print("Schema %s is valid JSON" % filename)
    except SchemaError:
        raise
        sys.exit(1)
    if base_uri:
        resolver = RefResolver(base_uri=base_uri,
                               referrer=filename)
    else:
        resolver = None
    return Draft7Validator(schema=schema,
                           resolver=resolver) 

def do_inputs():
    parser=argparse.ArgumentParser(description="Check assumptions of image pipeline inputs.",formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('input_format',help="Specify the type of input data you want to read")
    parser.add_argument('--panel',help="a json file with the panel definition",required=True)
    args = parser.parse_args()
    return args

def entry_point():
    args = do_inputs()

_schema_dir = os.path.split(os.path.realpath(__file__))[0]

#import json, os

# Get the directory where our schemas are located
#panel = json.loads(open(os.path.join(__schema_dir,'panel.json'),'r').read())
#report = json.loads(open(os.path.join(__schema_dir,'report.json'),'r').read())
