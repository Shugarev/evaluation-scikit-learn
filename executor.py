import warnings
def warn(*args, **kwargs):
    pass
warnings.warn = warn

from dataset_tester import DatasetTester
from model_creator import ModelCreator

from settings import Params
from utils import ParserArgs

parser = ParserArgs()

if Params.TASK == 'Teacher':
    model_creator = ModelCreator()
    model_creator.create_model(Params.ALGORITHM, Params.INPUT_PATH, Params.ALGORITHM_PARAMS, Params.MODEL_PATH)

if Params.TASK in ['Tester','TesterOrder']:
    dataset_tester = DatasetTester()
    if Params.TASK == 'TesterOrder' and Params.THRESHOLD:
        dataset_tester.threshold = Params.THRESHOLD

    dataset_tester.test_dataset(Params.INPUT_PATH, Params.MODEL_PATH, Params.OUTPUT_PATH, Params.ALGORITHM, Params.ANALYZER_PATH, Params.TASK)

# TODO подбор параметров для алгоритма
if Params.TASK == 'Selection_parameters':
    combinations = ParserArgs.get_combinations(Params.PARAMS_RANGE)
    model_creator = ModelCreator()
    model_creator.find_best_params(Params.INPUT_PATH,Params.ALGORITHM_PARAMS, Params.MODEL_PATH, Params.ALGORITHM, Params.PARAMS_RANGE)
    dataset_tester = DatasetTester()
    task = 'Tester'
    for algorithm_params in combinations:
        print(algorithm_params)
        model_creator.create_model(Params.ALGORITHM, Params.INPUT_PATH, algorithm_params, Params.MODEL_PATH)
        description = Params.MODEL_PATH.split('/')[-1] + str(algorithm_params)
        dataset_tester.test_dataset(Params.INPUT_PATH_2, Params.MODEL_PATH,Params.OUTPUT_PATH, Params.ALGORITHM, Params.ANALYZER_PATH, task, description=description)
