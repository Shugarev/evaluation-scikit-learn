#!/usr/bin/env python
import warnings


def warn( *args, **kwargs ):
    pass


warnings.warn = warn
from dataset_tester import datasetTester
from model_creator import modelCreator
from settings import Params
from utils import ParserArgs
from utils import get_combinations
from dataset_preprocessing import encoder

if __name__ == '__main__':
    parser = ParserArgs()

    if Params.TASK == 'Teacher':
        modelCreator.run(Params.INPUT_PATH, Params.ALGORITHM, Params.ALGORITHM_PARAMS, Params.MODEL_PATH)

    if Params.TASK in ['Tester', 'TesterOrder']:
        if Params.TASK == 'TesterOrder' and Params.THRESHOLD:
            datasetTester.set_threshold(Params.THRESHOLD)
        datasetTester.run(Params.TASK, Params.INPUT_PATH, Params.ALGORITHM, Params.MODEL_PATH,
                           Params.OUTPUT_PATH, Params.ANALYZER_PATH)

    if Params.TASK == 'Selection_parameters':
        combinations = get_combinations(Params.PARAMS_RANGE)
        modelCreator.find_best_params(Params.INPUT_PATH, Params.ALGORITHM_PARAMS, Params.ALGORITHM,
                                       Params.PARAMS_RANGE)
        task = 'Tester'
        for algorithm_params in combinations:
            print(algorithm_params)
            modelCreator.run(Params.INPUT_PATH, Params.ALGORITHM, algorithm_params, Params.MODEL_PATH)
            description = Params.MODEL_PATH.split('/')[-1] + str(algorithm_params)
            datasetTester.run(task, Params.INPUT_PATH_2, Params.ALGORITHM, Params.MODEL_PATH,
                               Params.OUTPUT_PATH, Params.ANALYZER_PATH, description=description)

    if Params.TASK in ['Encode_teach', 'Encode_test']:
        encoder.run(Params.TASK, Params.INPUT_PATH, Params.ENCODE_PARAMS, Params.ENCODED_PATH)
