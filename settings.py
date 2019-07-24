import os


class Params:

    # Build paths inside the project like this: os.path.join(BASE_DIR, ...)
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    ALGORITHMS = ['adaboost', 'xgboost', 'gausnb', 'decisiontree','gradientboost', 'logregression', 'linear_sgd']
    TASKS = ['Teacher', 'Tester', 'TesterOrder', 'Selection_parameters']
    TASK = None
    INPUT_PATH = None
    INPUT_PATH_2 = None
    MODEL_PATH = None
    OUTPUT_PATH = None
    ANALYZER_PATH = None
    THRESHOLD = None
    ALGORITHM = None
    ALGORITHM_PARAMS = {}
    PARAMS_RANGE = {}
    CONFIG_PATH = None
    BAD_STATUSES = ('true', 1, '1')
    MESSAGE = """ Required format
    python executor.py  --task Teacher --input db_teach.csv --algorithm_name adaboost --model_name ada_model --algorithm_config config-adaboost.ini
    python executor.py  --task Tester  --input db_test.csv  --algorithm_name adaboost --model_name ada_model --output Test-result-adaboost.csv
    python executor.py  --task TesterOrder  --input db_test1b.csv  --algorithm_name adaboost --model_name ada_model --threshold 0.50813 --output Test-result-adaboost-1.csv
    python executor.py  --task Selection_parameters --input db_teach.csv --algorithm_name adaboost --model_name ada_model --algorithm_config config-adaboost.ini --output Test-result-adabust.csv --input2 db_test.csv
    """
