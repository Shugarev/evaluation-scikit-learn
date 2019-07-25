import argparse
import configparser
import sys
from itertools import product
from typing import List, Dict

from settings import Params


class ParserArgs:

    def __init__( self, argv: Dict = None ):
        self.argv = argv or sys.argv
        p = argparse.ArgumentParser()
        p.add_argument('--task', default=None)
        p.add_argument('--algorithm_name', default=None)
        p.add_argument('--model_name', default=None)
        p.add_argument('--input', default=None)
        p.add_argument('--input2', default=None)
        p.add_argument('--algorithm_config', default=None)
        p.add_argument('--output', default=None)
        p.add_argument('--threshold', default=None)
        p.add_argument('--encode_config', default=None)
        p.add_argument('--encoded_path', default=None)
        self.args = p.parse_args()
        self.parse_args_from_comand_line()
        task = self.args.task
        if task in ['Teacher', 'Tester', 'TesterOrder', 'Selection_parameters']:
            self.set_algorithm_params()
        elif task in ['Encode_teach', 'Encode_test']:
            self.set_encode_params()

    def check_required_arguments( self ):
        args = self.args
        if args.task not in Params.TASKS:
            raise AttributeError(
                "Option 'task' is required. It is enum type {}. \n".format(str(Params.TASKS)) + Params.MESSAGE)
        if not args.input:
            raise AttributeError("Option 'input' is required.\n" + Params.MESSAGE)
        if args.task in ['Teacher', 'Tester', 'TesterOrder', 'Selection_parameters'] and not args.model_name:
            raise AttributeError("Option 'model_name' is required.\n" + Params.MESSAGE)
        if args.task == "Teacher" and args.algorithm_name not in Params.ALGORITHMS:
            raise AttributeError(
                "Option 'algorithm_name' is enum type {} .\n".format(str(Params.ALGORITHMS)) + Params.MESSAGE)
        if args.task in ['Tester', 'TesterOrder']:
            if not args.output:
                raise AttributeError("Option 'output' for Tester is requared.\n" + Params.MESSAGE)

    def parse_args_from_comand_line( self ):
        self.check_required_arguments()
        args = self.args
        Params.TASK = args.task
        Params.INPUT_PATH = Params.BASE_DIR + '/datasets/' + args.input
        if Params.TASK in ['Teacher', 'Tester', 'TesterOrder', 'Selection_parameters']:
            Params.MODEL_PATH = Params.BASE_DIR + '/models/' + args.model_name
            Params.ALGORITHM = args.algorithm_name
        if Params.TASK in ['Tester', 'TesterOrder', 'Selection_parameters']:
            Params.OUTPUT_PATH = Params.BASE_DIR + '/results/' + args.output
            Params.ANALYZER_PATH = Params.OUTPUT_PATH.split('.')[0] + "-analyzer.csv"
        if Params.TASK == 'TesterOrder':
            Params.THRESHOLD = args.threshold
        if Params.TASK == 'Selection_parameters':
            Params.INPUT_PATH_2 = Params.BASE_DIR + '/datasets/' + args.input2
        if Params.TASK in ['Encode_teach', 'Encode_test']:
            Params.ENCODED_PATH = Params.BASE_DIR + '/configs/' + args.encoded_path

    def get_config( self ):
        args = self.args
        if args.algorithm_config or args.encode_config:
            if args.algorithm_config:
                config_path = Params.CONFIG_PATH = Params.BASE_DIR + '/configs/' + args.algorithm_config
            elif args.encode_config:
                config_path = Params.ENCODE_CONFIG_PATH = Params.BASE_DIR + '/configs/' + args.encode_config
            config = configparser.ConfigParser()
            config.sections()
            config.read(config_path)
            return config
        return None

    def get_algorithm_params( self, config: Dict ) -> Dict:
        algorithm_params = {}
        if config:
            if "NUMERIC_PARAMS" in config.sections():
                algorithm_params = {k: float(v) if "." in v else int(v) for k, v in config["NUMERIC_PARAMS"].items()}
            if "STRING_PARAMS" in config.sections():
                algorithm_params = {**algorithm_params, **config["STRING_PARAMS"]}
        return algorithm_params

    def get_params_range( self, config: Dict ) -> Dict:
        options = {**config['PARAMS_RANGE']}
        return {k: eval(v) for k, v in options.items()}

    def set_algorithm_params( self ):
        config = self.get_config()
        if Params.TASK in ['Teacher', 'Selection_parameters']:
            Params.ALGORITHM_PARAMS = self.get_algorithm_params(config)

        if Params.TASK == 'Selection_parameters':
            Params.PARAMS_RANGE = self.get_params_range(config)

    def set_encode_params( self ):
        config = self.get_config()
        if 'SKIP_COLUMNS' in config.sections():
            options = {**config['SKIP_COLUMNS']}
            if 'skip_columns' in options:
                Params.ENCODE_PARAMS = {k: eval(v) for k, v in options.items()}


def get_combinations( options: Dict ) -> List[Dict]:
    """
    :param options:
        {
            n_estimators = [50, 60, 80, 100, 120]
            learning_rate = [0.2, 0.5, 0.7, 1.0]
            random_state = [123]
        }
    :return: [{'n_estimators': 50,'learning_rate':0.2,'random_state':123}, {...},...,{...}]
    """
    keys = options.keys()
    values = (options[key] for key in keys)
    combinations = [dict(zip(keys, combination)) for combination in product(*values)]
    return combinations
