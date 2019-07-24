import os
from typing import List, Dict, Tuple


class Params:
    # Build paths inside the project like this: os.path.join(BASE_DIR, ...)
    BASE_DIR:str = os.path.dirname(os.path.abspath(__file__))
    ALGORITHMS: List[str] = ['adaboost', 'xgboost', 'gausnb', 'decisiontree', 'gradientboost', 'logregression', 'linear_sgd']
    TASKS: List[str] = ['Teacher', 'Tester', 'TesterOrder', 'Selection_parameters']
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
    BAD_STATUSES: Tuple[str, int, str] = ('true', 1, '1')
    MESSAGE: str = """ Required format
    python executor.py  --task Teacher --input db_teach.csv --algorithm_name adaboost --model_name ada_model --algorithm_config config-adaboost.ini
    python executor.py  --task Tester  --input db_test.csv  --algorithm_name adaboost --model_name ada_model --output Test-result-adaboost.csv
    python executor.py  --task TesterOrder  --input db_test1b.csv  --algorithm_name adaboost --model_name ada_model --threshold 0.50813 --output Test-result-adaboost-1.csv
    python executor.py  --task Selection_parameters --input db_teach.csv --algorithm_name adaboost --model_name ada_model --algorithm_config config-adaboost.ini --output Test-result-adabust.csv --input2 db_test.csv
    """
