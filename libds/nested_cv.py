import numpy as np
import optuna
from sklearn.metrics import confusion_matrix
optuna.logging.set_verbosity(optuna.logging.WARNING)
from sklearn.feature_selection import RFE
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
import xgboost

from .metrics.stats import get_stats_from_ypred

class NestedCV:
    # TODO: Switch to control Stratified or not
    # TODO: Switch to control class balance

    def __init__(
            self,
            trials=3500,
            timeout=60*60,
            inner_splits=3,
            outer_splits=4,
            scoring='roc_auc',
            verbose=0
            ):
        self.trials = trials
        self.timeout = timeout
        self.inner_splits = inner_splits
        self.outer_splits = outer_splits
        self.scoring = scoring
        self.verbose = verbose
    
    def _cross_validate(self, model, X, y):
        """
        Runs crossvalidation with model on X, y
        """
        cv = StratifiedKFold(n_splits=self.inner_splits)
        results = cross_validate(model, X, y,
                    n_jobs=-1, cv=cv, scoring=self.scoring,
                    return_estimator=True, return_train_score=True
                    )
        
        score = results['test_score'].mean()

        return score, results


    def _create_rfe_xgb(self, params):
        """
        Creates a pipeline with RFE and XGBClassifier.
        """
        params = params.copy()
        rfe_params = dict()
        for param in [param for param in params if 'rfe__' in param]:
            rfe_params[param.replace('rfe__', '')] = params[param]
            del params[param]
        
        xgb = xgboost.XGBClassifier(**params)
        rfe  = RFE(estimator=xgb, **rfe_params)
        pipeline = Pipeline(steps=[('s', rfe), ('m', xgb)])

        return pipeline
    

    def _convert_param_definition_to_optuna_trial(self, trial, param_definition):
        """
        {'param1', 'int', min_value, max_value, False} => trial.suggest_int('param1', min_value, max_value)
        """

        # TODO: int should accept logarithmic parameter

        if param_definition[1] == 'float':
            optuna_trial_func = trial.suggest_float(
                param_definition[0], param_definition[2], param_definition[3], log=param_definition[4])
        elif param_definition[1] == 'int':
            optuna_trial_func = trial.suggest_int(
                param_definition[0], param_definition[2], param_definition[3], log=param_definition[4])
        elif param_definition[1] == 'categorical':
            optuna_trial_func = trial.suggest_categorical(
                param_definition[0], param_definition[2])
        else:
            raise ValueError(f"Unknown parameter type: {param_definition[1]}")

        return optuna_trial_func
        

    def _optuna_objective(self, trial, X, y, create_model_func, params_definitions):
        """
        """
        params = dict()
        for param_definition in params_definitions:
            params[param_definition[0]] = self._convert_param_definition_to_optuna_trial(trial, param_definition)
        
        pipeline = create_model_func(params)
        
        score, results = self._cross_validate(pipeline, X, y)

        return score


    def _find_best_params_with_optuna(self, create_model_func, params_definitions, X, y):
        """
        """

        study = optuna.create_study(direction='maximize')
        study.optimize(
            lambda trial: self._optuna_objective(
                trial, X, y, create_model_func, params_definitions
            ),
            n_trials=self.trials,
            timeout=self.timeout,
            show_progress_bar=False,
            n_jobs=-1
        )
        
        return study.best_params, len(study.trials)

    def _get_best_features(self, pipeline, feature_names):
        # TODO: this will break with other pipelines
        if feature_names:
            return np.array(feature_names)[pipeline.named_steps['s'].support_]
        return pipeline.named_steps['s'].support_
    

    def nested_cv(self, create_model_func, params_definitions, X, y, feature_names=None):
        """
        """
        all_stats = list()
        cms       = list()
        info      = list()
        
        cv = StratifiedKFold(n_splits=self.outer_splits)
        for n, (train_val_idx, test_idx) in enumerate(cv.split(X, y)):
            if self.verbose > 0:
                print(f"Fold {n+1}/{self.outer_splits}")

            X_train_val, X_test = X[train_val_idx], X[test_idx]
            y_train_val, y_test = y[train_val_idx], y[test_idx]

            best_params, total_trials = self._find_best_params_with_optuna(
                create_model_func, params_definitions, X_train_val, y_train_val
            )

            pipeline = create_model_func(best_params)
            pipeline.fit(X_train_val, y_train_val)

            stats, cm = self.compute_stats(pipeline, X_test, y_test)
            all_stats.append(stats)
            cms.append(cm)
            info.append(
                dict(
                    best_params      = best_params,
                    total_trials     = total_trials,
                    best_features    = self._get_best_features(pipeline, feature_names)
                )
            )

        return dict(
            info  = info,
            stats = all_stats,
            cms   = cms,
        )
    

    def compute_stats(self, pipeline, X_test, y_test):
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:,1]
        cm = confusion_matrix(y_test, y_pred)
        stats = get_stats_from_ypred(y_test, y_pred, y_proba)

        return stats, cm