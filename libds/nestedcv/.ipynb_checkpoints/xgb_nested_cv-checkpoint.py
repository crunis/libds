# TODO
# - Collect training and validation metrics

import sklearn
import xgboost as xgb
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import RFE

from sklearn.model_selection import cross_validate, RepeatedStratifiedKFold, RepeatedKFold

import optuna
import numpy as np

import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score, \
  average_precision_score, brier_score_loss, log_loss

def get_stats_from_cm(cm, digits=None, add_percentage_sign=False):
    tn, fp, fn, tp \
           = [item for row in cm for item in row]
    n0     = tn+fn
    n1     = tp+fp
    n      = n0+n1
    sens   = tp / (tp + fn) if (tp+fn) else float('nan')
    spec   = tn / (tn + fp) if (tn+fp) else float('nan')
    lr_pos = sens / (1 - spec) if (1-spec) else float('nan')
    lr_neg = (1 - sens) / spec if (spec)   else float('nan')
    dor    = lr_pos / lr_neg  if lr_neg else float('nan') # Diagnostic odd ratio
    ppv    = tp/(tp+fp) if (tp+fp)>0 else float('nan')

    res = {
        'n'      : n0+n1,
        'P%'     : (fn + tp)*100/n, # prevalence
        'n0'     : tn+fn,
        'n1'     : tp+fp,
        'n0%'    : n0*100/n,
        'n1%'    : n1*100/n,
        'FN'     : fn,
        'TP'     : tp,
        'FN%'    : fn*100/n0 if n0 else float('nan'),
        'TP%'    : tp*100/n1 if n1 else float('nan'),
        'sens%'  : sens*100, # also called recall
        'spec%'  : spec*100,
        'ppv%'   : 100*tp/(tp+fp) if (tp+fp) else float('nan'), # also called precission
        'npv%'   : 100*tn/(tn+fn) if (tn+fn) else float('nan'),
        'lr_pos' : lr_pos,
        'lr_neg' : lr_neg,
        'dor'    : dor,
        'acc'    : (tp+tn) / (tp+tn+fp+fn),
        'f1'     : 2*ppv*sens/(ppv + sens) if (ppv + sens) else float('nan'),
    }

    if digits is not None:
        for k,v in res.items():
            if add_percentage_sign and k[-1]=='%':
                res[k] = f"{v:.1f}%"
            else:
                res[k] = round(v, digits)

    return res


def get_stats_from_ypred(
    y_test,
    y_pred,
    y_scores=None,
    digits=None,
    add_percentage_sign=False,
    thresholds=None,
):
    cm = confusion_matrix(y_test, y_pred)
    stats = get_stats_from_cm(cm, digits=digits, add_percentage_sign=add_percentage_sign)

    if y_scores is not None:
        auc = roc_auc_score(y_test, y_scores)
        auc_pr = average_precision_score(y_test, y_scores)
        stats |= {'auc': auc, 'auc-pr': auc_pr}

        # Briers
        brier_score         = brier_score_loss(y_test, y_scores)
        stats['brier']      = brier_score
        brier_ref           = y_test.mean()*(1-y_test.mean())
        if brier_ref == 0:
            brier_scal = 0 if brier_score == 0 else -np.inf
        else:
            brier_scal = 1 - (brier_score / brier_ref)
        stats['brier_scal'] = brier_scal

        # LogLosses
        y_test_clase_1 = y_test[y_test == 1]
        probas_clase_1 = y_scores[y_test == 1]

        y_test_clase_0 = y_test[y_test == 0]
        probas_clase_0 = y_scores[y_test == 0]

        logloss_clase_1 = log_loss(y_test_clase_1, probas_clase_1, labels=[0, 1])
        logloss_clase_0 = log_loss(y_test_clase_0, probas_clase_0, labels=[0, 1])

        stats['logloss']    = log_loss(y_test, y_scores)
        stats['logloss_0']  = logloss_clase_0
        stats['logloss_1']  = logloss_clase_1

    if thresholds is not None:
        if y_scores is None:
            raise ValueError('Cant use multiple thresholds without y_scores')

        threshold_stats = dict()
        for threshold in thresholds:
            y_pred = (y_scores>=threshold)
            cm = confusion_matrix(y_test, y_pred)
            threshold_stats[threshold] = get_stats_from_cm(cm, digits=digits, add_percentage_sign=add_percentage_sign)

        return threshold_stats

    return stats



class XGBNestedCV:
    def __init__(
            self,
            inner_jobs=-1,
            outer_jobs=-1,
            inner_splits=3,
            inner_repeats=1,
            outer_splits=4,
            outer_repeats=1,
            stratified=False,
            ### OPTUNA
            trials = 3,
            timeout = 60*60, # 1 hour
            ### Model
            max_features = 10,
            step = 1,
            balance_classes = False,
            scoring='roc_auc',
            ):
        self.inner_jobs = inner_jobs
        self.outer_jobs = outer_jobs
        self.inner_splits = inner_splits
        self.inner_repeats = inner_repeats
        self.outer_splits = outer_splits
        self.outer_repeats = outer_repeats
        self.trials = trials
        self.timeout = timeout
        self.max_features = max_features
        self.step = step
        self.balance_classes = balance_classes
        self.scoring = scoring

        if stratified:
            self.cv = RepeatedStratifiedKFold
        else:
            self.cv = RepeatedKFold



    def create_rfe_xgb(self, xgb_params, rfe_params):
        xgb_rfe = xgb.XGBClassifier(
            **xgb_params
        )
        xgb_eval = sklearn.base.clone(xgb_rfe)

        rfe = RFE(estimator=xgb_rfe, **rfe_params)
        pipeline = Pipeline(steps=[('s', rfe), ('m', xgb_eval)])

        return pipeline


    def objective(
        self,
        trial,
        X,
        y,
        feature_names,
        ):
        n_features        = trial.suggest_int(   'n_features', 5, self.max_features)
        lr                = trial.suggest_float( 'learning_rate', 0.1, 0.3, log=False)
        max_depth         = trial.suggest_int(   'max_depth', 2, 10)
        subsample         = trial.suggest_float( 'subsample', 0.5, 1)
        colsample_bytree  = trial.suggest_float( 'colsample_bytree', 0.5, 1)
        n_estimators      = trial.suggest_int(   'n_estimators', 10, 1000, log=True)
        reg_alpha         = trial.suggest_float( 'reg_alpha', 0.001, 1, log=True)
        reg_lambda        = trial.suggest_float( 'reg_lambda', 0.001, 10, log=True)
        gamma             = trial.suggest_float( 'gamma', 0, 1)
        min_child_weight  = trial.suggest_float( 'min_child_weight', 0.1, 10, log=True)

        pipeline = self.create_rfe_xgb(
            dict(
                objective='binary:logistic',
                learning_rate = lr,
                max_depth = max_depth,
                subsample = subsample,
                colsample_bytree = colsample_bytree,
                n_estimators = n_estimators,
                reg_alpha = reg_alpha,
                reg_lambda = reg_lambda,
                gamma = gamma,
                min_child_weight = min_child_weight,
                scale_pos_weight = self._calculate_scale_weight(y)
            ),
            dict(
                n_features_to_select = n_features,
                step = self.step
            )
        )

        cv = self.cv(n_splits=self.inner_splits, n_repeats=self.inner_repeats)
        results = cross_validate(pipeline, X, y,
                            n_jobs=self.inner_jobs, cv=cv, scoring=self.scoring,
                            return_estimator=True, return_train_score=True
                            )

        final_score = results['test_score'].mean()

        if final_score > self.best_score:
            print(f"Best Score: {final_score:.4f}")
            print(f"Best n_features: {n_features}")
            best_features_now = np.array(feature_names)[
                results['estimator'][np.argmax(results['test_score'])].named_steps['s'].support_
            ]
            print(', '.join(best_features_now))

            self.best_score         = final_score
            self.last_best_features = best_features_now
            self.train_score        = results['train_score'].mean()

        return final_score
    
    
    def reset_stats(self):
        self.total_trials = 0
        self.all_info     = list()
        self.all_stats    = list()
        self.all_stats_threshold = list()
        self.cms          = list()
        
    
    def _calculate_scale_weight(self, y):
        """Calculates scale_pos_weight for imbalanced datasets."""
        if self.balance_classes:
            return (len(y) - np.sum(y)) / np.sum(y)
        return 1.0

    
    def compute_stats(self, pipeline, X_test, y_test, best_hyperparams):
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:,1]
        stats = get_stats_from_ypred(y_test, y_pred, y_proba)
#         stats['val_score'] = validation_score
        self.all_stats.append(stats)

        stats_threshold = get_stats_from_ypred(y_test, y_pred, y_proba, thresholds = [0.1, 0.25, 0.75, 0.9])
        self.all_stats_threshold.append(stats_threshold)

        self.cms.append(confusion_matrix(y_test, y_pred))

        self.all_info.append(dict(
          best_features    = self.last_best_features.tolist(),
          best_params      = best_hyperparams,
          total_trials     = self.total_trials,
        ))
        
    ###########################################
    ## go(). Does Nested CV
    ###########################################
    def go(self, X, y, feature_names):
        self.reset_stats()

        outer_cv = self.cv(n_splits=self.outer_splits, n_repeats=self.outer_repeats)
        for train_val_idx, test_idx in outer_cv.split(X, y):
            X_train_val, X_test = X[train_val_idx], X[test_idx]
            y_train_val, y_test = y[train_val_idx], y[test_idx]

            self.best_score = 0
            study = optuna.create_study(direction='maximize')
            study.optimize(
                lambda trial: self.objective(
                    trial, X_train_val, y_train_val, feature_names
                ),
                n_trials=self.trials,
                timeout=self.timeout,
                show_progress_bar=True,
                n_jobs=self.outer_jobs
            )
            self.total_trials += len(study.trials)
            validation_score = study.best_value
            best_hyperparams = study.best_params.copy()
            
            best_hyperparams_xgb = best_hyperparams.copy()
            best_hyperparams_xgb['scale_pos_weight'] = self._calculate_scale_weight(y_train_val)
            n_features = best_hyperparams_xgb.pop('n_features')

            print("DONE. Computing final model and stats ...")

            # Fit Model with full train_val and best hyperparams
            pipeline = self.create_rfe_xgb(
                best_hyperparams_xgb,
                dict(
                    n_features_to_select=n_features,
                    step = self.step
                )
            )
            pipeline.fit(X_train_val, y_train_val)

            self.compute_stats(pipeline, X_test, y_test, best_hyperparams)

        data = dict(
                  info   = self.all_info,
                  stats  = self.all_stats,
                  stats_threshold = self.all_stats_threshold,
                  cms    = self.cms,
              )
            
        print(f"Total trials: {self.total_trials}")
        return data
    
    
    def final(self, X, y, feature_names):
        self.best_score = 0
        study = optuna.create_study(direction='maximize')
        study.optimize(
            lambda trial: self.objective(
                trial, X, y, feature_names
            ),
            n_trials=self.trials,
            timeout=self.timeout,
            show_progress_bar=True,
            n_jobs=self.outer_jobs
        )

        validation_score = study.best_value
        best_hyperparams = study.best_params.copy()

        best_hyperparams_xgb = best_hyperparams.copy()
        best_hyperparams_xgb['scale_pos_weight'] = self._calculate_scale_weight(y)
        n_features = best_hyperparams_xgb.pop('n_features')

        print("DONE. Computing final model and stats ...")

        # Fit Model with full train_val and best hyperparams
        pipeline = self.create_rfe_xgb(
            best_hyperparams_xgb,
            dict(
                n_features_to_select=n_features,
                step = self.step
            )
        )
        pipeline.fit(X, y)

        return pipeline
