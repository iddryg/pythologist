import json
from importlib.resources import files
from pythologist.schemas import get_validator

report_definition_schema_validator = get_validator(files('schema_data.inputs').joinpath('report_definition.json'))
report_schema_validator = get_validator(files('schema_data.inputs').joinpath('report.json'))

def convert_report_definition_to_report(report_definition_json):
    """
    Take a report_definition and return a report in valid json format
    """
    report_definition_schema_validator.validate(report_definition_json)

    output = json.loads(json.dumps(report_definition_json)) # get deep copy of the definition to modify.. parameters will come directly from that
    # Do the regions
    for i,measure in enumerate(output['region_selection']):
        # always should have at least one mutually exclusive phenotype so no worries about Nonetype
        output['region_selection'][i]['regions_to_combine'] = [x.strip() for x in measure['regions_to_combine'].split(',')]

    # Start with the mutually exclusive phenotypes
    for i,measure in enumerate(output['population_densities']):
        # always should have at least one mutually exclusive phenotype so no worries about Nonetype
        output['population_densities'][i]['mutually_exclusive_phenotypes'] = [x.strip() for x in measure['mutually_exclusive_phenotypes'].split(',')]
        # binary could be empty if it is set it to no members
        if measure['binary_phenotypes'] is None:
            output['population_densities'][i]['binary_phenotypes'] = []
        else:
            output['population_densities'][i]['binary_phenotypes'] = \
                [{'target_name':x.strip()[:-1].strip(),'filter_direction':x.strip()[-1]} for x in measure['binary_phenotypes'].split(',')]
    for i,measure in enumerate(output['population_percentages']):
        # always should have at least one mutually exclusive phenotype so no worries about Nonetype
        output['population_percentages'][i]['numerator_mutually_exclusive_phenotypes'] = [x.strip() for x in measure['numerator_mutually_exclusive_phenotypes'].split(',')]
        # binary could be empty if it is set it to no members
        if measure['numerator_binary_phenotypes'] is None:
            output['population_percentages'][i]['numerator_binary_phenotypes'] = []
        else:
            output['population_percentages'][i]['numerator_binary_phenotypes'] = \
                [{'target_name':x.strip()[:-1].strip(),'filter_direction':x.strip()[-1]} for x in measure['numerator_binary_phenotypes'].split(',')]
        # always should have at least one mutually exclusive phenotype so no worries about Nonetype
        output['population_percentages'][i]['denominator_mutually_exclusive_phenotypes'] = [x.strip() for x in measure['denominator_mutually_exclusive_phenotypes'].split(',')]
        # binary could be empty if it is set it to no members
        if measure['denominator_binary_phenotypes'] is None:
            output['population_percentages'][i]['denominator_binary_phenotypes'] = []
        else:
            output['population_percentages'][i]['denominator_binary_phenotypes'] = \
                [{'target_name':x.strip()[:-1].strip(),'filter_direction':x.strip()[-1]} for x in measure['denominator_binary_phenotypes'].split(',')]

    report_schema_validator.validate(output)
    return output
