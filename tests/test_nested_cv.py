import pandas as pd
import sklearn.datasets

from libds.nested_cv import NestedCV

def test_nested_cv():
    import collections

    X, y = sklearn.datasets.load_iris(return_X_y=True)
    X = X[50:,:]
    y = y[50:]
    y = y - 1

    import xgboost as xgb

    clf = xgb.XGBClassifier(random_state = 1111)
    ncv = NestedCV(clf, {}, random_state = 1111, use_tqdm=False)

    ncv.run(X, y)

    viv = pd.DataFrame(ncv.stats_).describe().loc[['mean', 'std'], ['sens%', 'spec%', 'auc', 'auc-pr']]

    assert round(viv['auc']['mean'],   4) == 0.9845
    assert round(viv['auc']['std'],    4) == 0.0076
    assert round(viv['sens%']['mean'], 4) == 91.9118
    assert round(viv['sens%']['std'],  4) == 7.0143
    assert round(viv['spec%']['mean'], 4) == 94.1176
    assert round(viv['spec%']['std'],  4) == 5.8824


def test_nested_cv_tune_threshold():
    import collections

    X, y = sklearn.datasets.load_iris(return_X_y=True)
    X = X[50:,:]
    y = y[50:]
    y = y - 1

    import xgboost as xgb

    clf = xgb.XGBClassifier(random_state = 1111)
    ncv = NestedCV(clf, {}, random_state = 1111, use_tqdm=False, tune_threshold=True)

    ncv.run(X, y)

    viv = pd.DataFrame(ncv.stats_).describe().loc[['mean', 'std'], ['sens%', 'spec%', 'auc', 'auc-pr']]

    assert round(viv['auc']['mean'],   4) == 0.9845
    assert round(viv['auc']['std'],    4) == 0.0076
    assert round(viv['sens%']['mean'], 4) == 93.9951
    assert round(viv['sens%']['std'],  4) == 0.2123
    assert round(viv['spec%']['mean'], 4) == 92.1569
    assert round(viv['spec%']['std'],  4) == 6.7924


def test_nested_cv_w_imbalanced_data():
    X, y = sklearn.datasets.load_iris(return_X_y=True)
    # Trying to make it hard for the estimator
    #   1. classes 1 (versicolor) and 2 (virginica) overlap
    #   2. Mostly overlap on sepal stuff (0, 1)
    X = X[50:-25,0:2]
    y = y[50:-25]

    y = y - 1

    import xgboost as xgb

    clf = xgb.XGBClassifier(random_state = 1111)
    ncv = NestedCV(clf, {}, random_state = 1111, use_tqdm=False)

    ncv.run(X, y)

    viv = pd.DataFrame(ncv.stats_).describe().loc[['mean', 'std'], ['sens%', 'spec%', 'auc', 'auc-pr']]

    assert round(viv['auc']['mean'],   4) == 0.6219
    # assert round(viv['auc']['std'],    4) == 0.0076
    assert round(viv['sens%']['mean'], 4) == 28.7037
    # assert round(viv['sens%']['std'],  4) == 7.0143
    assert round(viv['spec%']['mean'], 4) == 80.3922
    # assert round(viv['spec%']['std'],  4) == 5.8824


def test_nested_cv_w_imbalanced_data_tune_threshold():
    X, y = sklearn.datasets.load_iris(return_X_y=True)
    # Trying to make it hard for the estimator
    #   1. classes 1 (versicolor) and 2 (virginica) overlap
    #   2. Mostly overlap on sepal stuff (0, 1)
    X = X[50:-25,0:2]
    y = y[50:-25]

    y = y - 1

    import xgboost as xgb

    clf = xgb.XGBClassifier(random_state = 1111)
    ncv = NestedCV(clf, {}, random_state = 1111, use_tqdm=False, tune_threshold=True)

    ncv.run(X, y)

    viv = pd.DataFrame(ncv.stats_).describe().loc[['mean', 'std'], ['sens%', 'spec%', 'auc', 'auc-pr']]

    assert round(viv['auc']['mean'],   4) == 0.6219
    # assert round(viv['auc']['std'],    4) == 0.0076
    assert round(viv['sens%']['mean'], 4) == 32.4074
    # assert round(viv['sens%']['std'],  4) == 7.0143
    assert round(viv['spec%']['mean'], 4) == 78.1863
    # assert round(viv['spec%']['std'],  4) == 5.8824