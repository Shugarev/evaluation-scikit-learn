import numpy as np
import pandas as pd

def replace_na(dataset, replace_val=-9999):
    return dataset.fillna(replace_val)

def get_sample_weight(dataset):
    n_sample = dataset.shape[0]
    n_bad = dataset[dataset.status.isin(['true',1])].shape[0]
    weight_bad = min(int(n_sample / n_bad), 99)
    n = n_bad * weight_bad + n_sample - n_bad
    sample_weight = np.where(dataset.status.isin(['true',1]), weight_bad/n, 1/n)
    return sample_weight

def save_feature_importances(teach, drop_columns, feature_importances, feature_path):
    importances = pd.DataFrame({"feature_name": teach.drop(drop_columns, axis=1, errors="ignore").columns,
                                "importances": feature_importances})
    importances.sort_values(by=["importances"], ascending=False, inplace=True)
    importances.to_csv(feature_path, index=False, quoting=1)