import os
from typing import List, Dict, Tuple


class Params:
    # Build paths inside the project like this: os.path.join(BASE_DIR, ...)
    BASE_DIR:str = os.path.dirname(os.path.abspath(__file__))
    ALGORITHMS: List[str] = ['adaboost', 'xgboost', 'gausnb', 'decisiontree', 'gradientboost', 'logregression', 'linear_sgd']
    TASKS: List[str] = ['Teacher', 'Tester', 'TesterOrder', 'Selection_parameters', 'Encode_teach', 'Encode_test']
    TASK = None
    INPUT_PATH:str = None
    INPUT_PATH_2:str = None
    MODEL_PATH:str = None
    OUTPUT_PATH:str = None
    ANALYZER_PATH:str = None
    THRESHOLD:float = None
    ALGORITHM:str = None
    ALGORITHM_PARAMS: Dict = {}
    PARAMS_RANGE: Dict = {}
    CONFIG_PATH:str = None
    ENCODED_PATH: str = None
    ENCODE_CONFIG_PATH: str = None
    ENCODE_PARAMS: Dict = {}
    BAD_STATUSES: Tuple[str, int, str] = ('true', 1, '1')
    MESSAGE: str = """ Required format
    python executor.py --task Encode_teach --input db_teach-bayes.csv --encode_config config-bayes-encode.ini  --encoded_path encoded_bayes
    python executor.py --task Encode_test  --input db_test-bayes.csv  --encode_config config-bayes-encode.ini  --encoded_path encoded_bayes
    python executor.py --task Teacher --input db_teach.csv --algorithm_name adaboost --model_name ada_model --algorithm_config config-adaboost.ini
    python executor.py --task Tester  --input db_test.csv  --algorithm_name adaboost --model_name ada_model --output Test-result-adaboost.csv
    python executor.py --task TesterOrder  --input db_test1b.csv  --algorithm_name adaboost --model_name ada_model --threshold 0.50813 --output Test-result-adaboost-1.csv
    python executor.py --task Selection_parameters --input db_teach.csv --algorithm_name adaboost --model_name ada_model --algorithm_config config-adaboost.ini --output Test-result-adabust.csv --input2 db_test.csv
    """
    REQUIRED_ARGS = {
        'Default': ['task', 'input'],
        'Teacher': ['algorithm_name', 'model_name'],
        'Tester': ['algorithm_name', 'model_name', 'output'],
        'TesterOrder': ['algorithm_name', 'model_name', 'output', 'threshold'],
        'Selection_parameters': ['algorithm_name', 'model_name', 'algorithm_config', 'input2', 'output'],
        'Encode_teach': ['encode_config', 'encoded_path'],
        'Encode_test': ['encode_config', 'encoded_path']
    }

    @staticmethod
    def get_all_args():
        all_args = []
        for v in Params.REQUIRED_ARGS:
            all_args += v
        return set(all_args)