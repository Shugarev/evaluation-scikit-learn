import pandas as pd
from settings import Params

def get_empty_prediction_df(metric):
    result_df = pd.DataFrame(
        columns=['description', 'p_1', 'p_2', 'p_3', 'p_5'
            , 'p_6', 'p_10', 'p_20', 'p_50'
            , 'test', 'cb_test'])
    col_names = ['test', 'cb_test']
    result_df.loc[:, col_names] = result_df.loc[:, col_names].astype(int)

    col_names = [col for col in result_df.columns if col.startswith('p_')]
    col_names.append('description')
    result_df.loc[:, col_names] = result_df.loc[:, col_names].astype(str)
    if metric == 'amount':
        result_df['cb_test_amount'] = 0
        result_df['test_amount'] = 0

    return result_df


def get_df_prediction(Test_1, result_df=None, metric="count", description='teach - test'):
    if type(result_df).__name__ != 'DataFrame':
        result_df = get_empty_prediction_df(metric)

    test = Test_1.copy()
    All_cb_in_test = test[test['status'].isin(Params.BAD_STATUSES)].shape[0]
    All_row_in_test = test.shape[0]

    test["probability"] = pd.to_numeric(test.probability, errors="coerce")
    test.sort_values(by="probability", ascending=False, inplace=True)
    row = {'description': description
        , 'test': All_row_in_test
        , 'cb_test': All_cb_in_test
           }
    if metric == "amount":
        test["amount"] = pd.to_numeric(test.amount, errors="coerce")
        test_amount = round(sum(test.amount), 2)
        cb_test_amount = round(sum(test[test['status'].isin(Params.BAD_STATUSES)].amount), 2)
        row['cb_test_amount'] = cb_test_amount
        row['test_amount'] = test_amount
        test["cum_amount"] = test.amount.cumsum()

    row_threshold = row.copy()
    row_threshold['description'] = 'threshold'
    for col in result_df:
        if col.startswith('p_'):
            d = int(col.split('_')[1])
            if metric == "amount":
                dt = test[test.cum_amount < d * test_amount / 100]
                dt_cb_amount = sum(dt[dt.status.isin(Params.BAD_STATUSES) ].amount)
                row[col] = str(round(100 * dt_cb_amount / cb_test_amount, 2))
                n = dt.shape[0]
            else:
                n = round(All_row_in_test * d / 100)
                dt_p = test.iloc[:n, :]
                n_cb = dt_p[dt_p.status.isin(Params.BAD_STATUSES)].shape[0]
                row[col] = str(round(100 * n_cb / All_cb_in_test, 2))
            row_threshold[col] = str(round(test.probability.values[n - 1], 6))

    result_df = result_df.append([row, row_threshold], ignore_index=True)
    return result_df