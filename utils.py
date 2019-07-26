import argparse
import sys
from configparser import ConfigParser
from itertools import product
from typing import List, Dict

from settings import Params


class ParserArgs:

    def __init__(self, argv: Dict = None):
        self.argv = argv or sys.argv
        p = argparse.ArgumentParser()
        for arg in Params.get_all_args():
            p.add_argument(arg, default=None)
        self.args = p.parse_args()
        self._set_params()
        task = self.args.task
        if task in ['Teacher', 'Tester', 'TesterOrder', 'Selection_parameters']:
            self._set_algorithm_params()
        elif task in ['Encode_teach', 'Encode_test']:
            self._set_encode_params()

    def _set_params(self):
        self._check_required_arguments()
        args = self.args
        Params.TASK = args.task
        Params.INPUT_PATH = Params.BASE_DIR + '/datasets/' + args.input

        Params.ENCODED_PATH = Params.BASE_DIR + '/configs/' + args.encoded_path if args.encoded_path else None
        Params.ENCODE_CONFIG_PATH = Params.BASE_DIR + '/configs/' + args.encode_config if args.encode_config else None

        Params.MODEL_PATH = Params.BASE_DIR + '/models/' + args.model_name if args.model_name else None
        Params.ALGORITHM = args.algorithm_name if args.algorithm_name else None
        Params.CONFIG_PATH = Params.BASE_DIR + '/configs/' + args.algorithm_config if args.algorithm_config else None

        Params.OUTPUT_PATH = Params.BASE_DIR + '/results/' + args.output if args.output else None
        Params.ANALYZER_PATH = Params.OUTPUT_PATH.rsplit('.', maxsplit=1)[0] + "-analyzer.csv" if args.output else None

        Params.INPUT_PATH_2 = Params.BASE_DIR + '/datasets/' + args.input2 if args.input2 else None
        Params.THRESHOLD = args.threshold if args.threshold else None


    def _check_required_arguments(self):
        args = {k: v for k, v in vars(self.args).items() if v}
        required_args = Params.REQUIRED_ARGS
        if not all(elem in args.keys() for elem in required_args['Default']):
            raise AttributeError(
                "Options {} are required. \n".format(str(required_args['Default'])) + Params.MESSAGE)
        task = args['task']
        if task not in Params.TASKS:
            raise ValueError(
                "Option 'task' is enum type {}. \n".format(str(Params.TASKS)) + Params.MESSAGE)
        if not all(elem in args.keys() for elem in required_args[task]):
            raise AttributeError(
                "Options {} are required. \n".format(str(required_args[task])) + Params.MESSAGE)
        if task == "Teacher" and args['algorithm_name'] not in Params.ALGORITHMS:
            raise ValueError(
                "Option 'algorithm_name' is enum type {} .\n".format(str(Params.ALGORITHMS)) + Params.MESSAGE)

    def _set_algorithm_params(self):
        config = self._get_config()
        if Params.TASK in ['Teacher', 'Selection_parameters']:
            Params.ALGORITHM_PARAMS = self._get_algorithm_params(config)
        if Params.TASK == 'Selection_parameters':
            Params.PARAMS_RANGE = self._get_params_range(config)

    def _set_encode_params(self):
        config = self._get_config()
        if 'SKIP_COLUMNS' in config.sections():
            options = {**config['SKIP_COLUMNS']}
            if 'skip_columns' in options:
                Params.ENCODE_PARAMS = {k: eval(v) for k, v in options.items()}

    def _get_config(self) -> ConfigParser:
        args = self.args
        if args.algorithm_config or args.encode_config:
            config_path = None
            if args.algorithm_config:
                config_path = Params.CONFIG_PATH
            elif args.encode_config:
                config_path = Params.ENCODE_CONFIG_PATH
            config = ConfigParser()
            config.sections()
            config.read(config_path)
            return config
        return None

    def _get_algorithm_params(self, config: ConfigParser) -> Dict:
        algorithm_params = {}
        if config:
            if "NUMERIC_PARAMS" in config.sections():
                algorithm_params = {k: float(v) if "." in v else int(v) for k, v in config["NUMERIC_PARAMS"].items()}
            if "STRING_PARAMS" in config.sections():
                algorithm_params = {**algorithm_params, **config["STRING_PARAMS"]}
        return algorithm_params

    def _get_params_range(self, config: ConfigParser) -> Dict:
        options = {**config['PARAMS_RANGE']}
        return {k: eval(v) for k, v in options.items()}

def get_combinations(options: Dict) -> List[Dict]:
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
