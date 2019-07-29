import sys
import unittest
import pandas as pd

from dataset_tester import DatasetTester
from model_creator import ModelCreator
from settings import Params
from utils import ParserArgs
from dataset_preprocessing import Encoder

teacher_args = ['--task', 'Teacher', '--input', 'db_teach.csv', '--algorithm_name'
    , 'adaboost', '--model_name', 'ada_model', '--algorithm_config', 'config-adaboost.ini']

tester_args = ['--task', 'Tester', '--input', 'db_test.csv', '--algorithm_name', 'adaboost'
    , '--model_name', 'ada_model', '--output', 'Test-result-adaboost.csv']

tester_order_args = ['--task', 'TesterOrder', '--input', 'db_test1c.csv', '--algorithm_name', 'adaboost'
            , '--model_name', 'ada_model', '--threshold', '0.505484', '--output', 'Test-result-adaboost-1.csv']

encode_teach_args  = ['--task', 'Encode_teach', '--input', 'db_teach-bayes.csv', '--encode_config'
    , 'config-bayes-encode.ini', '--encoded_path', 'encoded_bayes']

encode_test_args = ['--task', 'Encode_test', '--input', 'db_test-bayes.csv', '--encode_config'
    , 'config-bayes-encode.ini', '--encoded_path', 'encoded_bayes']
class TestAlgorithmsParams(unittest.TestCase):


    def test_params_ada_task_teach(self):
        copy_argv = sys.argv.copy()
        sys.argv[1:] = teacher_args
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
        sys.argv[1:] = tester_args
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
        sys.argv[1:] = tester_order_args
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

    def test_params_bayes_task_encode_teach(self):
        copy_argv = sys.argv.copy()
        sys.argv[1:] = encode_teach_args
        ParserArgs()
        message = 'Param {} is incorrect.'
        self.assertEqual(Params.TASK, 'Encode_teach', message.format('TASK'))
        self.assertEqual(Params.INPUT_PATH, Params.BASE_DIR + '/datasets/db_teach-bayes.csv', message.format('INPUT_PATH'))
        self.assertEqual(Params.ENCODED_PATH, Params.BASE_DIR +  '/configs/encoded_bayes', message.format('ENCODED_PATH'))
        self.assertEqual(Params.ENCODE_CONFIG_PATH, Params.BASE_DIR + '/configs/config-bayes-encode.ini', message.format('ENCODE_CONFIG_PATH'))
        self.assertDictEqual(Params.ENCODE_PARAMS,  {'skip_columns': ['status', 'bin', 'amount_deviation', 'client_hour', 'day_of_week']}, message.format('ENCODE_PARAMS'))
        sys.argv = copy_argv

    def test_params_bayes_task_encode_test(self):
        copy_argv = sys.argv.copy()
        sys.argv[1:] = encode_test_args
        ParserArgs()
        message = 'Param {} is incorrect.'
        self.assertEqual(Params.TASK, 'Encode_test', message.format('TASK'))
        self.assertEqual(Params.INPUT_PATH, Params.BASE_DIR + '/datasets/db_test-bayes.csv', message.format('INPUT_PATH'))
        self.assertEqual(Params.ENCODED_PATH, Params.BASE_DIR +  '/configs/encoded_bayes', message.format('ENCODED_PATH'))
        self.assertEqual(Params.ENCODE_CONFIG_PATH, Params.BASE_DIR + '/configs/config-bayes-encode.ini', message.format('ENCODE_CONFIG_PATH'))
        self.assertDictEqual(Params.ENCODE_PARAMS,  {'skip_columns': ['status', 'bin', 'amount_deviation', 'client_hour', 'day_of_week']}, message.format('ENCODE_PARAMS'))
        sys.argv = copy_argv


class TestAlgorithmsresult(unittest.TestCase):

    def test_ada_result(self):
        copy_argv = sys.argv.copy()
        sys.argv[1:] = teacher_args
        ParserArgs()
        model_creator = ModelCreator()
        model_creator.run(Params.INPUT_PATH, Params.ALGORITHM, Params.ALGORITHM_PARAMS, Params.MODEL_PATH)

        sys.argv[1:] = tester_args
        ParserArgs()
        dataset_tester = DatasetTester()
        dataset_tester.run(Params.TASK, Params.INPUT_PATH, Params.ALGORITHM, Params.MODEL_PATH,
                           Params.OUTPUT_PATH, Params.ANALYZER_PATH)

        sys.argv[1:] = tester_order_args
        ParserArgs()
        dataset_tester = DatasetTester()
        dataset_tester.set_threshold(Params.THRESHOLD)
        dataset_tester.run(Params.TASK, Params.INPUT_PATH, Params.ALGORITHM, Params.MODEL_PATH,
                           Params.OUTPUT_PATH, Params.ANALYZER_PATH)
        sys.argv = copy_argv

        df = pd.read_csv(Params.BASE_DIR  + '/results/Test-result-adaboost-analyzer.csv', dtype=str).to_dict()
        df_expected = pd.read_csv(Params.BASE_DIR  + '/data_for_tests/Test-result-adaboost-analyzer.csv', dtype=str).to_dict()
        self.assertDictEqual(df, df_expected, 'Model work is incorrect')

        df = pd.read_csv(Params.BASE_DIR  + '/results/Test-result-adaboost-1.csv', dtype=str).to_dict()
        df_expected = pd.read_csv(Params.BASE_DIR + '/data_for_tests/Test-result-adaboost-1.csv', dtype=str).to_dict()
        self.assertDictEqual(df, df_expected, 'Test one order is correct')
        sys.argv = copy_argv

class Test_Encoder(unittest.TestCase):

    def test_encode_teach_test(self):
        copy_argv = sys.argv.copy()
        sys.argv[1:] = encode_teach_args
        ParserArgs()
        encoder = Encoder()
        encoder.run(Params.TASK, Params.INPUT_PATH, Params.ENCODE_PARAMS, Params.ENCODED_PATH)
        sys.argv[1:] = encode_test_args
        ParserArgs()
        encoder = Encoder()
        encoder.run(Params.TASK, Params.INPUT_PATH, Params.ENCODE_PARAMS, Params.ENCODED_PATH)

        df = pd.read_csv(Params.BASE_DIR + '/datasets/db_teach-bayes-encoded.csv', dtype=str).to_dict()
        df_expected = pd.read_csv(Params.BASE_DIR + '/data_for_tests/db_teach-bayes-encoded.csv',
                                  dtype=str).to_dict()
        self.assertDictEqual(df, df_expected, 'Teach is encoded incorrect')

        df = pd.read_csv(Params.BASE_DIR + '/datasets/db_test-bayes-encoded.csv', dtype=str).to_dict()
        df_expected = pd.read_csv(Params.BASE_DIR + '/data_for_tests/db_test-bayes-encoded.csv',
                                  dtype=str).to_dict()
        self.assertDictEqual(df, df_expected, 'Teach is encoded incorrect')
        sys.argv = copy_argv


if __name__ == '__main__':
    unittest.main()
