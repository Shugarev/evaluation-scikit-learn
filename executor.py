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

if args.type == "Teacher" and args.algorithm_name not in ['adaboost', 'xgboost', 'gausnb', 'decisiontree', 'gradientboost', 'logregression']:
    raise BaseException("Option 'type' is requared. It must be choosen from 'adaboost', 'xgboost', 'gausnb', 'decisiontree', 'gradientboost', 'logregression'.\n" + message)

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
        if "NUMERIC_PARAMS" in config.sections():
            algorithm_params = {k: float(v) if "." in v else int(v) for k,v in config["NUMERIC_PARAMS"].items()}
        if "STRING_PARAMS" in config.sections():
            algorithm_params = {**algorithm_params, **config["STRING_PARAMS"]}
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

#TODO подбор параметров для алгоритма
if type == 'selection_parameters':
    model_creator = ModelCreator()
    model_creator.find_best_params(args.algorithm_name, input_path, algorithm_params, model_path)
    parameters = {
        'n_estimators': [60, 50, 80],
        'learning_rate': [0.7, 0.2, 1.0]
    }
    for n in parameters.get('n_estimators'):
        for l in  parameters.get('learning_rate'):
            algorithm_params = {'n_estimators':n, 'learning_rate':l, 'random_state':123}
            input_path= '/home/sergey/PycharmProjects/evaluation-scikit-learn/datasets/db_teach.csv'
            model_creator = ModelCreator()
            model_creator.create_model(args.algorithm_name, input_path, algorithm_params, model_path)

            output_path = BASE_DIR + '/results/ada.csv'
            analyzer_path = output_path.split('.')[0] + "-analyzer.csv"
            type = 'Tester'
            input_path='/home/sergey/PycharmProjects/evaluation-scikit-learn/datasets/db_test.csv'
            dataset_tester = DatasetTester()
            dataset_tester.test_dataset(input_path, model_path, output_path, args.algorithm_name, analyzer_path, type)
            print ('n= ', n, 'l= ', l)
            type = 'Teacher'