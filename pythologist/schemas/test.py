import unittest, os, json,sys
from pythologist_schemas import get_validator

class TestValidSchemas(unittest.TestCase):
    pass

class TestExampleSchemas(unittest.TestCase):
    pass

def create_file_exists_test(filename):
    # Pre: Take a filename path of an expected file
    # Post: Return True is True if the file exists
    result = os.path.exists(filename)
    def do_test(self):
        self.assertTrue(result)
    return do_test

def create_valid_json_test(filename):
    # Pre: Take a filename path
    # Post: Return True is true if the filename validates as JSON
    result = False
    try:
        if os.path.exists(filename):
            with open(filename,'rt') as f:
                data = f.read()
                json.loads(data)
                result = True
    except:
        result = False
    def do_test(self):
        self.assertTrue(result)
    return do_test

def create_valid_schema_format_test(filename, schemas_dir):
    # Pre: Take a schema filename
    # Post: Return a test that passes as True is True when the schema file is properly formatted json-schema
    result = False
    data = None
    try:
        if os.path.exists(filename):
            with open(filename,'rt') as f:
                data = f.read()
                data = json.loads(data)
                result = True
        validator = get_validator(filename,schemas_dir)
        result = True
    except:
        result = False
    def do_test(self):
        self.assertTrue(result)
    return do_test

def create_validated_example_test(example_filename, schema_filename, schemas_dir):
    # Pre: Take the example schema, the schema, and the schemas_dir 
    # Post: Return a test that passes as True is True when the example is a valid version of the Schema
    result = False
    data = None
    try:
        if os.path.exists(example_filename) and os.path.exists(schema_filename):
            with open(schema_filename,'rt') as f:
                data = f.read()
                data = json.loads(data)
            validator = get_validator(schema_filename,schemas_dir)
            with open(example_filename,'rt') as f:
                data = f.read()
                data = json.loads(data)
            result = True
            try:
                 validator.validate(data)
            except:
                result = False
    except:
        result = False
    def do_test(self):
        self.assertTrue(result)
    return do_test

schemas_dir = os.path.join('pythologist_schemas','schemas')
examples_dir = os.path.join('pythologist_schemas','json_examples')
for root,dirs,files in os.walk(schemas_dir):
    for f in files:
        if f[-5:] != '.json': continue
        base = root[len(schemas_dir)+1:]
        schema_path = os.path.join(schemas_dir,base,f)
        example_path = os.path.join(examples_dir,base,f)

        test_method = create_file_exists_test(example_path)
        test_method.__name__ = 'testing example exists: '+example_path
        setattr(TestExampleSchemas,test_method.__name__,test_method)
        test_method = create_valid_json_test(example_path)
        test_method.__name__ = 'testing if example is valid json: '+example_path
        setattr(TestExampleSchemas,test_method.__name__,test_method)

        test_method = create_valid_json_test(schema_path)
        test_method.__name__ = 'testing if schema file is valid json: '+schema_path
        setattr(TestValidSchemas,test_method.__name__,test_method)

        test_method = create_valid_schema_format_test(schema_path,'file://'+os.path.abspath(schemas_dir)+'/')
        test_method.__name__ = 'testing if schema file is valid json-schema: '+schema_path
        setattr(TestValidSchemas,test_method.__name__,test_method)

        test_method = create_validated_example_test(example_path,schema_path,'file://'+os.path.abspath(schemas_dir)+'/')
        test_method.__name__ = 'testing if example file is validated example of the json-schema: '+example_path
        setattr(TestExampleSchemas,test_method.__name__,test_method)


if __name__ == '__main__':
    unittest.main()