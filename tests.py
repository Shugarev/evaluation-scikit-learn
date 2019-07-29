import sys
import unittest

from settings import Params
from utils import ParserArgs


class TestAlgorithmsParams(unittest.TestCase):

    def test_params_ada_task_teach(self):
        copy_argv = sys.argv.copy()
        sys.argv[1:] = ['--task', 'Teacher', '--input', 'db_teach.csv', '--algorithm_name'
            , 'adaboost', '--model_name', 'ada_model', '--algorithm_config', 'config-adaboost.ini']
        ParserArgs()
        message = 'Param {} is incorrect.'
        self.assertEqual(Params.INPUT_PATH, Params.BASE_DIR + '/datasets/db_teach.csv'
                         , message.format('input path'))
        self.assertEqual(Params.TASK, 'Teacher', message.format('TASK'))

        self.assertEqual(Params.ALGORITHM, 'adaboost', message.format('ALGORITHM'))
        self.assertDictEqual(Params.ALGORITHM_PARAMS, {'n_estimators': 140, 'learning_rate': 1, 'random_state': 123}
                         , message.format('ALGORITHM_PARAMS'))
        sys.argv = copy_argv

    def test_params_ada_task_test(self):
        copy_argv = sys.argv.copy()
        sys.argv[1:] = ['--task', 'Tester', '--input', 'db_test.csv', '--algorithm_name', 'adaboost'
            , '--model_name', 'ada_model', '--output', 'Test-result-adaboost.csv']

        ParserArgs()
        message = 'Param {} is incorrect.'
        self.assertEqual(Params.INPUT_PATH, Params.BASE_DIR + '/datasets/db_test.csv'
                         , message.format('input path'))
        self.assertEqual(Params.TASK, 'Tester', message.format('TASK'))

        self.assertEqual(Params.ALGORITHM, 'adaboost', message.format('ALGORITHM'))
        self.assertDictEqual(Params.ALGORITHM_PARAMS, {'n_estimators': 140, 'learning_rate': 1, 'random_state': 123}
                         , message.format('ALGORITHM_PARAMS'))

        self.assertEqual(Params.MODEL_PATH, Params.BASE_DIR + '/models/ada_model', message.format('MODEL_PATH'))
        sys.argv = copy_argv

    def test_params_ada_task_TesterOrder(self):
        copy_argv = sys.argv.copy()
        sys.argv[1:] =['--task', 'TesterOrder', '--input', 'db_test1c.csv', '--algorithm_name', 'adaboost'
            , '--model_name', 'ada_model', '--threshold', '0.505484', '--output', 'Test-result-adaboost-1.csv']

        ParserArgs()
        message = 'Param {} is incorrect.'
        self.assertEqual(Params.INPUT_PATH, Params.BASE_DIR + '/datasets/db_test1c.csv'
                         , message.format('input path'))
        self.assertEqual(Params.TASK, 'TesterOrder', message.format('TASK'))

        self.assertEqual(Params.ALGORITHM, 'adaboost', message.format('ALGORITHM'))
        self.assertEqual(Params.MODEL_PATH, Params.BASE_DIR + '/models/ada_model', message.format('MODEL_PATH'))
        self.assertEqual(Params.OUTPUT_PATH, Params.BASE_DIR + '/results/Test-result-adaboost-1.csv'
                         , message.format('OUTPUT_PATH'))

        self.assertEqual('{:.6f}'.format(float(Params.THRESHOLD)), '{:.6f}'.format(0.505484), message.format('THRESHOLD'))
        sys.argv = copy_argv



if __name__ == '__main__':
    unittest.main()
