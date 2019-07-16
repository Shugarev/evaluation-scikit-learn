import numpy as np
import os
import argparse
import configparser

from model_creator import ModelCreator

# Build paths inside the project like this: os.path.join(BASE_DIR, ...)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

p = argparse.ArgumentParser()
p.add_argument('--type', default=None)
p.add_argument('--algorithm_name', default=None)
p.add_argument('--model_name', default=None)
p.add_argument('--input', default=None)
p.add_argument('--algorithm_config', default=None)
args = p.parse_args()

message = """ Required format
python executor --type Teacher --input datasets/db_teach.csv --algorithm_name adaboost --model_name models/ada_model --algorithm_config configs/config-adaboost.ini
python executor --type Tester  --input datasets/db_test.csv  --algorithm_name adaboost --model_name models/ada_model 
"""

if args.type not in ['Teacher', 'Tester']:
    raise BaseException("Option 'type' is requared. It must be choose from Teacher or Tester.\n" + message)

if not args.input:
    raise BaseException("Option 'input' is requared.\n" + message)

if not args.model_name:
    raise BaseException("Option 'model_name' is requared.\n" + message)

type = args.type
input_path = BASE_DIR + '/' + args.input
model_path = BASE_DIR + '/' + args.model_name

if type == 'Teacher':
    algorithm_params = {}
    if args.algorithm_config:
        config_path = BASE_DIR + '/' + args.algorithm_config
        config = configparser.ConfigParser()
        config.sections()
        config.read(args.algorithm_config)
        algorithm_params = {
            key: float(config["DEFAULT"][key]) if "." in config["DEFAULT"][key] else int(config["DEFAULT"][key])
            for key in config["DEFAULT"]}
    model_creator = ModelCreator()
    model_creator.create_model(args.algorithm_name, input_path, algorithm_params, model_path)
