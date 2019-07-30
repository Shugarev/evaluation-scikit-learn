import joblib
import numpy as np
import pandas as pd
from sklearn import preprocessing

from settings import Params


def replace_na( dataset: pd.DataFrame, replace_val=-9999 ) -> pd.DataFrame:
    return dataset.fillna(replace_val)


def get_sample_weight( dataset: pd.DataFrame ) -> pd.Series:
    n_sample = dataset.shape[0]
    n_bad = dataset[dataset.status.isin(Params.BAD_STATUSES)].shape[0]
    weight_bad = min(int(n_sample / n_bad), 99)
    n = n_bad * weight_bad + n_sample - n_bad
    sample_weight = np.where(dataset.status.isin(Params.BAD_STATUSES), weight_bad / n, 1 / n)
    return sample_weight


def get_xgb_weight( dataset: pd.DataFrame ) -> pd.Series:
    n_sample = dataset.shape[0]
    n_bad = dataset[dataset.status.isin(Params.BAD_STATUSES)].shape[0]
    w = int(n_sample / n_bad)
    weight = np.where(dataset.status.isin(Params.BAD_STATUSES), w, 1)
    return weight


class encoder:


    @classmethod
    def run(cls, task, input_path, encode_params, encoded_path ):
        if task == 'Encode_teach':
            cls.encode_teach(input_path, encoded_path, encode_params.get('skip_columns'))
        else:
            cls.encode_test(input_path, encoded_path, encode_params.get('skip_columns'))

    @classmethod
    def encode_teach(cls, teach_path: str, encode_path: str, skip_columns: list = ['status'] ):
        teach_orig = pd.read_csv(teach_path, dtype=str)
        skip_columns = list(set(skip_columns) & set(list(teach_orig)))
        teach = teach_orig.drop(skip_columns, axis=1, errors="ignore")
        row = {key: teach.status.values[0] if key == "status" else 'unknown' for key in list(teach)}
        teach = teach.append(row, ignore_index=True)
        enc = preprocessing.OrdinalEncoder()
        enc.fit(teach)
        joblib.dump(enc, encode_path)
        teach = teach[:-1]
        cls.encode_dataset(teach_path, teach, enc, skip_columns, teach_orig)

    @classmethod
    def encode_test(cls, test_path: str, encode_path: str, skip_columns: list = ['status'] ):
        test_orig = pd.read_csv(test_path, dtype=str)
        skip_columns = list(set(skip_columns) & set(list(test_orig)))
        test = test_orig.drop(skip_columns, axis=1, errors="ignore")
        enc = joblib.load(encode_path)
        test = cls.replace_new_value_in_test_by_unknown(enc, test)
        cls.encode_dataset(test_path, test, enc, skip_columns, test_orig)

    @classmethod
    def encode_dataset(cls, dataset_path: str, dataset: pd.DataFrame, enc: list, skip_columns: list,
                        dataset_original: pd.DataFrame ):
        dataset_encode_path = dataset_path.rsplit('.', maxsplit=1)[0] + '-encoded.csv'
        dataset_encoded = enc.transform(dataset)
        dataset_encoded = pd.DataFrame(dataset_encoded)
        dataset_encoded = pd.concat([dataset_encoded, dataset_original[skip_columns]], axis=1)

        col_names = list(dataset) + skip_columns
        dataset_encoded.columns = col_names
        dataset_encoded = cls.encode_status_columns(dataset_encoded)
        dataset_encoded.to_csv(dataset_encode_path, index=False, quoting=1)

    @classmethod
    def encode_status_columns(cls, dataset: pd.DataFrame):
        df = dataset.copy()
        df.status = np.where(df.status.isin(Params.BAD_STATUSES), 1, 0)
        return df

    @classmethod
    def replace_new_value_in_test_by_unknown(cls, enc: list, dataset:pd.DataFrame):
        df = dataset.copy()
        for k, v in enumerate(enc.categories_):
            df.iloc[:, k] = np.where(df.iloc[:, k].isin(v), df.iloc[:, k], 'unknown')
        return df