import os
import argparse
import configparser

from model_creator import ModelCreator
from dataset_tester import DatasetTester

# Build paths inside the project like this: os.path.join(BASE_DIR, ...)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

p = argparse.ArgumentParser()
p.add_argument('--type', default=None)
p.add_argument('--algorithm_name', default=None)
p.add_argument('--model_name', default=None)
p.add_argument('--input', default=None)
p.add_argument('--algorithm_config', default=None)
p.add_argument('--output', default=None)
p.add_argument('--threshold', default=None)
args = p.parse_args()

message = """ Required format
python executor.py --type Teacher --input db_teach.csv --algorithm_name adaboost --model_name ada_model --algorithm_config config-adaboost.ini
python executor.py --type Tester  --input db_test.csv  --algorithm_name adaboost --model_name ada_model --output Test-result-adaboost.csv
python executor.py --type TesterOrder  --input db_test1a.csv --algorithm_name adaboost --model_name ada_model --output Test-result-adaboost1.csv --threshold 0.5001
"""

if args.type not in ['Teacher', 'Tester', 'TesterOrder']:
    raise BaseException("Option 'type' is requared. It must be choosen from 'Teacher', 'Tester' or 'TesterOrder'.\n" + message)

if args.type == "Teacher" and args.algorithm_name not in ['adaboost', 'xgboost', 'gausnb', 'decisiontree', 'gradientboost']:
    raise BaseException("Option 'type' is requared. It must be choosen from 'adaboost', 'xgboost', 'gausnb', 'decisiontree', 'gradientboost'.\n" + message)

if not args.input:
    raise BaseException("Option 'input' is requared.\n" + message)

if not args.model_name:
    raise BaseException("Option 'model_name' is requared.\n" + message)

type = args.type
input_path = BASE_DIR + '/datasets/' + args.input
model_path = BASE_DIR + '/models/' + args.model_name

if type == 'Teacher':
    if not args.algorithm_name:
        raise BaseException("Option 'algorithm_name' for Teacher is requared.\n" + message)
    algorithm_params = {}
    if args.algorithm_config:
        config_path = BASE_DIR + '/configs/' + args.algorithm_config
        config = configparser.ConfigParser()
        config.sections()
        config.read(config_path)
        algorithm_params = {
            key: float(config["DEFAULT"][key]) if "." in config["DEFAULT"][key] else int(config["DEFAULT"][key])
            for key in config["DEFAULT"]}
    model_creator = ModelCreator()
    model_creator.create_model(args.algorithm_name, input_path, algorithm_params, model_path)

if type == 'Tester' or type == 'TesterOrder':
    if not args.output:
        raise BaseException("Option 'output' for Tester is requared.\n" + message)
    output_path = BASE_DIR + '/results/' + args.output
    analyzer_path = output_path.split('.')[0] + "-analyzer.csv"
    dataset_tester = DatasetTester()
    if type == 'TesterOrder' and args.threshold:
        dataset_tester.threshold = args.threshold
    dataset_tester.test_dataset(input_path, model_path, output_path, args.algorithm_name, analyzer_path, type)