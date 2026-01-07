import numpy as np
import pandas as pd
from libds.nested_cv import NestedCV
import xgboost as xgb

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

def test_nested_cv():
    ncv = NestedCV()

    assert isinstance(ncv, NestedCV)


def test_cross_validate():
    from sklearn.linear_model import LogisticRegression
    from sklearn.datasets import make_classification
    
    X, y = make_classification(n_samples=1000, n_features=10, random_state=42)
    model1 = LogisticRegression()
    model2 = xgb.XGBClassifier()
    
    ncv = NestedCV()
    score1, _ = ncv._cross_validate(model1, X, y)
    score2, _ = ncv._cross_validate(model2, X, y)
    
    assert isinstance(score1, float)
    assert 0 <= score1 <= 1
    assert isinstance(score2, float)
    assert 0 <= score2<= 1
    assert score1 < score2

    # print(f"test_cross_validate scores: LogisticRegression {score1:.4f}, XGBClassifier {score2:.4f}")


def test_create_rfe_xgb():
    from sklearn.pipeline import Pipeline
    from sklearn.feature_selection import RFE
    import xgboost as xgb

    ncv = NestedCV()
    params = {
        'rfe__n_features_to_select': 5,
        'rfe__step': 1,
        'n_estimators': 100,
        'learning_rate': 0.1
    }
    pipeline = ncv._create_rfe_xgb(params.copy())

    assert isinstance(pipeline, Pipeline)
    assert isinstance(pipeline.named_steps['s'], RFE)
    assert isinstance(pipeline.named_steps['m'], xgb.XGBClassifier)
    assert pipeline.named_steps['s'].n_features_to_select == 5
    assert pipeline.named_steps['s'].step == 1
    assert pipeline.named_steps['m'].n_estimators == 100
    assert pipeline.named_steps['m'].learning_rate == 0.1


def test_convert_param_definition_to_optuna_trial():
    ncv = NestedCV()

    class MockTrial:
        def suggest_int(self, name, low, high, log=False):
            return f"int_{name}_{low}_{high}"
        def suggest_float(self, name, low, high, log=False):
            return f"float_{name}_{low}_{high}_{log}"
        def suggest_categorical(self, name, choices):
            return f"categorical_{name}_{choices}"
        def suggest_loguniform(self, name, low, high):
            return f"loguniform_{name}_{low}_{high}"
        def suggest_uniform(self, name, low, high):
            return f"uniform_{name}_{low}_{high}"
        def suggest_discrete_uniform(self, name, low, high, q):
            return f"discrete_uniform_{name}_{low}_{high}_{q}"
        def suggest_log_discrete_uniform(self, name, low, high, q):
            return f"log_discrete_uniform_{name}_{low}_{high}_{q}"

    mock_trial = MockTrial()

    # Test int
    param_def_int = ['param_int', 'int', 1, 10, False]
    result_int = ncv._convert_param_definition_to_optuna_trial(mock_trial, param_def_int)
    assert result_int == "int_param_int_1_10"

    # Test float
    param_def_float = ['param_float', 'float', 0.1, 0.9, True]
    result_float = ncv._convert_param_definition_to_optuna_trial(mock_trial, param_def_float)
    assert result_float == "float_param_float_0.1_0.9_True"

    # Test categorical
    param_def_categorical = ['param_cat', 'categorical', ['a', 'b', 'c']]
    result_categorical = ncv._convert_param_definition_to_optuna_trial(mock_trial, param_def_categorical)
    assert result_categorical == "categorical_param_cat_['a', 'b', 'c']"


def test_find_best_hyperparams_with_optuna():
    from sklearn.datasets import make_classification
    
    X, y = make_classification(n_samples=100, n_features=10, random_state=42)
    
    ncv = NestedCV(trials = 3, timeout = 1)
    
    # Define some parameter definitions for testing
    params_definitions = [
        ['n_estimators', 'int', 10, 50, False],
        ['learning_rate', 'float', 0.01, 0.1, True],
        ['rfe__n_features_to_select', 'int', 2, 5, False],
    ]
    
    best_params, total_trials = ncv._find_best_hyperparams_with_optuna(ncv._create_rfe_xgb, params_definitions, X, y)
    
    assert isinstance(best_params, dict)
    assert 'n_estimators' in best_params
    assert 'learning_rate' in best_params
    assert 'rfe__n_features_to_select' in best_params
    assert 10 <= best_params['n_estimators'] <= 50
    assert 0.01 <= best_params['learning_rate'] <= 0.1
    assert 2 <= best_params['rfe__n_features_to_select'] <= 5

    # print(f"Best params: {best_params}")


def test_nested_cv_full_run():
    from sklearn.datasets import make_classification
    
    X, y = make_classification(n_samples=200, n_features=10, random_state=42)
    
    ncv = NestedCV(trials=2, timeout=1) # Reduced trials and timeout for faster testing
    
    # Define parameter definitions for testing
    params_definitions = [
        ['n_estimators', 'int', 10, 20, False],
        ['learning_rate', 'float', 0.05, 0.1, True],
        ['rfe__n_features_to_select', 'int', 2, 3, False],
    ]
    
    res = ncv.nested_cv(ncv._create_rfe_xgb, params_definitions, X, y)
    all_stats = res['stats']
    
    assert isinstance(all_stats, list)
    assert len(all_stats) == 4 # Default n_splits for outer CV is 4
    
    for stats_dict in all_stats:
        assert isinstance(stats_dict, dict)
        assert 'auc' in stats_dict
        assert 'acc' in stats_dict
    
    df_stats = pd.DataFrame(all_stats).describe()

    assert df_stats.loc['mean', 'auc'] > 0.8
    assert df_stats.loc['mean', 'auc-pr'] > 0.7
    
    # print(f"All stats: {pd.DataFrame(all_stats).describe().to_dict()['auc']}")
    # print(f"All stats: {pd.DataFrame(all_stats).describe().to_dict()['auc-pr']}")
    

def test_get_cms():
    from sklearn.datasets import make_classification

    X, y = make_classification(n_samples=200, n_features=10, random_state=42)

    ncv = NestedCV(trials=2, timeout=1) # Reduced trials and timeout for faster testing

    # Define parameter definitions for testing
    params_definitions = [
        ['n_estimators', 'int', 10, 20, False],
        ['learning_rate', 'float', 0.05, 0.1, True],
        ['rfe__n_features_to_select', 'int', 2, 3, False],
    ]

    res = ncv.nested_cv(ncv._create_rfe_xgb, params_definitions, X, y)

    assert isinstance(res, dict)
    assert 'cms' in res
    assert isinstance(res['cms'], list)
    assert len(res['cms']) == 4 # Default n_splits for outer CV is 4

    assert res['cms'][0].shape == (2, 2)
    assert res['cms'][1].shape == (2, 2)
    assert res['cms'][2].shape == (2, 2)
    assert res['cms'][3].shape == (2, 2)

    # print(f"Confusion matrices: {res['cms']}")


def test_get_best_features():
    import numpy as np
    ncv = NestedCV()

    class MockStep:
        def __init__(self, support):
            self.support_ = np.array(support)

    class MockPipeline:
        def __init__(self, support):
            self.named_steps = {'s': MockStep(support)}

    support = [True, False, True]
    pipeline = MockPipeline(support)

    # Test 1: Return support boolean array when no feature names provided
    res = ncv._get_best_features(pipeline, None)
    np.testing.assert_array_equal(res, np.array(support))

    # Test 2: Return selected feature names
    feature_names = ['f1', 'f2', 'f3']
    res = ncv._get_best_features(pipeline, feature_names)
    np.testing.assert_array_equal(res, np.array(['f1', 'f3']))


def test_nested_cv_get_features():
    # TODO: is there a way to ensure all 3 first features are always choosen
    #       it keeps selecting only f1 and f2
    from sklearn.datasets import make_classification

    X, y = make_classification(
        n_samples=1000,
        n_features=6,
        n_informative=3,
        n_redundant=0,
        n_repeated=0,
        # 3 useless features
        random_state=42,
        scale = .125,
        shuffle = False
        )

    ncv = NestedCV(trials=10, timeout=1) # Reduced trials and timeout for faster testing

    # Define parameter definitions for testing
    params_definitions = [
        ['n_estimators', 'int', 10, 20, False],
        ['learning_rate', 'float', 0.05, 0.1, True],
        ['rfe__n_features_to_select', 'int', 2, 3, False],
    ]

    res = ncv.nested_cv(ncv._create_rfe_xgb, params_definitions, X, y, feature_names=['f1', 'f2', 'f3', 'f4', 'f5', 'f6'])

    assert isinstance(res, dict)
    assert 'info' in res
    assert isinstance(res['info'], list)
    assert len(res['info']) == 4
    assert 'best_features' in res['info'][0]
    assert isinstance(res['info'][0]['best_features'], np.ndarray)
    assert len(res['info'][0]['best_features']) <= 3 # Max features to select is 3
    assert 'f1' in res['info'][0]['best_features'] or \
           'f2' in res['info'][0]['best_features'] or \
           'f3' in res['info'][0]['best_features']

