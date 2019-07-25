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


def save_feature_importances( teach: pd.DataFrame, drop_columns: list, feature_importances: list, feature_path: str ):
    importances = pd.DataFrame({"feature_name": teach.drop(drop_columns, axis=1, errors="ignore").columns,
                                "importances": feature_importances})
    importances.sort_values(by=["importances"], ascending=False, inplace=True)
    importances.to_csv(feature_path, index=False, quoting=1)


def encode_teach( teach_path: str, encode_path: str, skip_columns: list = ['status'] ):
    teach_orig = pd.read_csv(teach_path, dtype=str)
    skip_columns = list(set(skip_columns) & set(list(teach_orig)))
    teach = teach_orig.drop(skip_columns, axis=1, errors="ignore")
    row = {key: teach.status.values[0] if key == "status" else 'unknown' for key in list(teach)}
    teach = teach.append(row, ignore_index=True)
    enc = preprocessing.OrdinalEncoder()
    enc.fit(teach)
    joblib.dump(enc, encode_path)
    teach = teach[:-1]
    encode_dataset(teach_path, teach, enc, skip_columns, teach_orig)


def encode_test( test_path: str, encode_path: str, skip_columns: list = ['status'] ):
    test_orig = pd.read_csv(test_path, dtype=str)
    skip_columns = list(set(skip_columns) & set(list(test_orig)))
    test = test_orig.drop(skip_columns, axis=1, errors="ignore")
    enc = joblib.load(encode_path)
    test = replace_new_value_in_test_by_unknown(enc, test)
    encode_dataset(test_path, test, enc, skip_columns, test_orig)


def encode_dataset( dataset_path: str, dataset: pd.DataFrame, enc: list, skip_columns: list,
                    dataset_original: pd.DataFrame ):
    dataset_encode_path = '.'.join(dataset_path.split('.')[0:-1]) + '-encoded.csv'
    dataset_encoded = enc.transform(dataset)
    dataset_encoded = pd.DataFrame(dataset_encoded)
    dataset_encoded = pd.concat([dataset_encoded, dataset_original[skip_columns]], axis=1)

    col_names = list(dataset) + skip_columns
    dataset_encoded.columns = col_names
    dataset_encoded = encode_status_columns(dataset_encoded)
    dataset_encoded.to_csv(dataset_encode_path, index=False, quoting=1)


def replace_new_value_in_test_by_unknown( enc, dataset ):
    df = dataset.copy()
    for k, v in enumerate(enc.categories_):
        df.iloc[:, k] = np.where(df.iloc[:, k].isin(v), df.iloc[:, k], 'unknown')
    return df


def encode_status_columns( dataset: pd.DataFrame ):
    df = dataset.copy()
    df.status = np.where(df.status.isin(Params.BAD_STATUSES), 1, 0)
    return df


class Encoder:
    def run( self, task, input_path, encode_params, encoded_path ):
        if task == 'Encode_teach':
            encode_teach(input_path, encoded_path, encode_params.get('skip_columns'))
        else:
            encode_test(input_path, encoded_path, encode_params.get('skip_columns'))
