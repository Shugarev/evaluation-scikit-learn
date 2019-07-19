import warnings
def warn(*args, **kwargs):
    pass
warnings.warn = warn


import argparse
import configparser
import os
from itertools import product

from dataset_tester import DatasetTester
from model_creator import ModelCreator



# Build paths inside the project like this: os.path.join(BASE_DIR, ...)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

p = argparse.ArgumentParser()
p.add_argument('--task', default=None)
p.add_argument('--algorithm_name', default=None)
p.add_argument('--model_name', default=None)
p.add_argument('--input', default=None)
p.add_argument('--input2', default=None)
p.add_argument('--algorithm_config', default=None)
p.add_argument('--output', default=None)
p.add_argument('--threshold', default=None)
args = p.parse_args()

message = """ Required format
python executor.py  --task Teacher --input db_teach.csv --algorithm_name adaboost --model_name ada_model --algorithm_config config-adaboost.ini
python executor.py  --task Tester  --input db_test.csv  --algorithm_name adaboost --model_name ada_model --output Test-result-adaboost.csv
python executor.py  --task TesterOrder  --input db_test1b.csv  --algorithm_name adaboost --model_name ada_model --threshold 0.50813 --output Test-result-adaboost-1.csv
python executor.py  --task Selection_parameters --input db_teach.csv --algorithm_name adaboost --model_name ada_model --algorithm_config config-adaboost.ini --output Test-result-adabust.csv --input2 db_test.csv
"""

if args.task not in ['Teacher', 'Tester', 'TesterOrder', 'Selection_parameters']:
    raise BaseException(
        "Option 'task' is required. It must be choosen from 'Teacher', 'Tester' or 'TesterOrder'.\n" + message)

if args.task == "Teacher" and args.algorithm_name not in ['adaboost', 'xgboost', 'gausnb', 'decisiontree',
                                                          'gradientboost', 'logregression']:
    raise BaseException(
        "Option 'task' is required. It must be choosen from 'adaboost', 'xgboost', 'gausnb', 'decisiontree', 'gradientboost', 'logregression'.\n" + message)

if not args.input:
    raise BaseException("Option 'input' is required.\n" + message)

if not args.model_name:
    raise BaseException("Option 'model_name' is required.\n" + message)

task = args.task
input_path = BASE_DIR + '/datasets/' + args.input
model_path = BASE_DIR + '/models/' + args.model_name

def get_config(args):
    if args.algorithm_config:
        config_path = BASE_DIR + '/configs/' + args.algorithm_config
        config = configparser.ConfigParser()
        config.sections()
        config.read(config_path)
        return config
    return None

def get_algorithm_params(config):
    algorithm_params = {}
    if config:
        if "NUMERIC_PARAMS" in config.sections():
            algorithm_params = {k: float(v) if "." in v else int(v) for k, v in config["NUMERIC_PARAMS"].items()}
        if "STRING_PARAMS" in config.sections():
            algorithm_params = {**algorithm_params, **config["STRING_PARAMS"]}
    return  algorithm_params


if task == 'Teacher':
    if not args.algorithm_name:
        raise BaseException("Option 'algorithm_name' for Teacher is required.\n" + message)
    config = get_config(args)
    algorithm_params = get_algorithm_params(config)
    model_creator = ModelCreator()
    model_creator.create_model(args.algorithm_name, input_path, algorithm_params, model_path)

if task == 'Tester' or task == 'TesterOrder':
    if not args.output:
        raise BaseException("Option 'output' for Tester is requared.\n" + message)
    output_path = BASE_DIR + '/results/' + args.output
    analyzer_path = output_path.split('.')[0] + "-analyzer.csv"
    dataset_tester = DatasetTester()
    if task == 'TesterOrder' and args.threshold:
        dataset_tester.threshold = args.threshold
    dataset_tester.test_dataset(input_path, model_path, output_path, args.algorithm_name, analyzer_path, task)

# TODO подбор параметров для алгоритма
if task == 'Selection_parameters':
    config = get_config(args)
    algorithm_params = get_algorithm_params(config)

    options = {**config['PARAMS_RANGE']}
    options = {k: eval(v) for k, v in options.items()}

    keys = options.keys()
    values = (options[key] for key in keys)
    combinations = [dict(zip(keys, combination)) for combination in product(*values)]

    model_creator = ModelCreator()
    model_creator.find_best_params(input_path, algorithm_params, model_path, args.algorithm_name, options)

    dataset_tester = DatasetTester()
    input_path_db_test = BASE_DIR + '/datasets/' + args.input2
    output_path = BASE_DIR + '/results/' + args.output
    analyzer_path = output_path.split('.')[0] + "-analyzer.csv"
    for algorithm_params in combinations:
        print(algorithm_params)
        task = 'Teacher'
        model_creator.create_model(args.algorithm_name, input_path, algorithm_params, model_path)
        task = 'Tester'
        description = model_path.split('/')[-1] + str(algorithm_params)
        dataset_tester.test_dataset(input_path_db_test, model_path, output_path, args.algorithm_name, analyzer_path,
                                    task, description=description)
