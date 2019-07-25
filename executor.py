#!/usr/bin/env python
import warnings


def warn( *args, **kwargs ):
    pass


warnings.warn = warn
from dataset_tester import DatasetTester
from model_creator import ModelCreator
from settings import Params
from utils import ParserArgs
from utils import get_combinations
from dataset_preprocessing import Encoder

if __name__ == '__main__':
    parser = ParserArgs()

    if Params.TASK == 'Teacher':
        model_creator = ModelCreator()
        model_creator.run(Params.INPUT_PATH, Params.ALGORITHM, Params.ALGORITHM_PARAMS, Params.MODEL_PATH)

    if Params.TASK in ['Tester', 'TesterOrder']:
        dataset_tester = DatasetTester()
        if Params.TASK == 'TesterOrder' and Params.THRESHOLD:
            dataset_tester.set_threshold(Params.THRESHOLD)
        dataset_tester.run(Params.TASK, Params.INPUT_PATH, Params.ALGORITHM, Params.MODEL_PATH,
                           Params.OUTPUT_PATH, Params.ANALYZER_PATH)

    if Params.TASK == 'Selection_parameters':
        combinations = get_combinations(Params.PARAMS_RANGE)
        model_creator = ModelCreator()
        model_creator.find_best_params(Params.INPUT_PATH, Params.ALGORITHM_PARAMS, Params.ALGORITHM,
                                       Params.PARAMS_RANGE)
        dataset_tester = DatasetTester()
        task = 'Tester'
        for algorithm_params in combinations:
            print(algorithm_params)
            model_creator.run(Params.INPUT_PATH, Params.ALGORITHM, algorithm_params, Params.MODEL_PATH)
            description = Params.MODEL_PATH.split('/')[-1] + str(algorithm_params)
            dataset_tester.run(task, Params.INPUT_PATH_2, Params.ALGORITHM, Params.MODEL_PATH,
                               Params.OUTPUT_PATH, Params.ANALYZER_PATH, description=description)

    if Params.TASK in ['Encode_teach', 'Encode_test']:
        encoder = Encoder()
        encoder.run(Params.TASK, Params.INPUT_PATH, Params.ENCODE_PARAMS, Params.ENCODED_PATH)
