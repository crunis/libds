# v1.0 20/10/2025
# Initial clean version

import numpy as np
from tqdm import tqdm
import sklearn
from sklearn.model_selection import RepeatedStratifiedKFold, TunedThresholdClassifierCV
from sklearn.metrics import confusion_matrix
from libds.metrics.stats import get_stats_from_ypred


def tune_threshold_process(model, X, y, splits=3, repeats=1, random_state=None, scoring='balanced_accuracy'):
    # This is used to add threshold functionality to any estimator
    import types 

    def predict_threshold(self, X, threshold = None):
        if threshold is None:
            threshold = self.threshold

        predictions = (self.predict_proba(X)[:,1]>threshold).astype(int)
        return predictions
    ###############################################################
    
    cv = RepeatedStratifiedKFold(
            n_splits     = splits, 
            n_repeats    = repeats, 
            random_state = random_state,
    )
    model = TunedThresholdClassifierCV(model, scoring=scoring, cv=cv)
    model.fit(X, y)
    best_threshold = model.best_threshold_

    model.predict = types.MethodType(predict_threshold, model)
    model.threshold = best_threshold

    return best_threshold, model


class NestedCV:
    def __init__(self, model, param_grid,
                 scoring                = 'average_precision',
                 repeats                = (1, 1),
                 n_splits               = (3, 3),
                 n_jobs                 = -1,
                 use_tqdm               = True,
                 tqdm_metrics           = ('auc-pr',),
                 verbose                = 1,
                 extra_info             = None,
                 tune_threshold         = False,
                 thresholds             = None,
                 tune_threshold_scoring = 'balanced_accuracy',
                 random_state           = None
                ):

        # --- Configuration ---
        self.model = model
        self.param_grid = param_grid
        self.scoring = scoring
        self.repeats = repeats
        self.n_splits = n_splits
        self.n_jobs = n_jobs
        self.use_tqdm = use_tqdm
        self.tqdm_metrics = tqdm_metrics
        self.verbose = verbose
        self.extra_info = extra_info or {}
        self.tune_threshold = tune_threshold
        self.tune_threshold_scoring = tune_threshold_scoring
        self.thresholds = thresholds or [0.5, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
        self.random_state   = random_state

        self.reset_results()


    def reset_results(self):
        self.models_ = []
        self.stats_ = []
        self.cms_ = []
        self.idxs_ = []
        self.results_ = {}    

        
    def tune_hyperparams(self, X_train_val, y_train_val):    
        cv = RepeatedStratifiedKFold(
                n_splits     = self.n_splits[1], 
                n_repeats    = self.repeats[1], 
                random_state = self.random_state,
        )
        clf = sklearn.model_selection.GridSearchCV(
                self.model,
                param_grid = self.param_grid,
                cv         = cv,
                n_jobs     = self.n_jobs,
                scoring    = self.scoring,
                verbose    = self.verbose,
                refit      = True,
        )
        
        return clf.fit(X_train_val, y_train_val), clf.best_estimator_
    

    def show_sens_spec(self, model, X, y):
        stats = get_stats_from_ypred(y, model.predict(X))
        print(f"sens: {stats['sens%']:.2f}  spec: {stats['spec%']:.2f}")

    
    def store_info(self, best_model, X_test, y_test, train_val_idx, test_idx):
        self.models_.append(best_model)
        y_pred = best_model.predict(X_test)
        y_proba = best_model.predict_proba(X_test)[:, 1]
        
        self.cms_.append(confusion_matrix(y_test, y_pred))
        stats = get_stats_from_ypred(y_test, y_pred, y_proba)
        self.stats_.append(stats)

        # Store indices
        tmp = np.column_stack([test_idx, y_pred])
        pred_idx = tmp[tmp[:, 1] == 1, 0].astype(int).tolist()
        self.idxs_.append({
            'train_val_idx': train_val_idx, 'test_idx': test_idx, 'pred_idx': pred_idx
        })

        return stats
    
    
    def run(self, X, y):
        """Executes the nested cross-validation process."""
        self.reset_results()
        
        outer_cv = RepeatedStratifiedKFold(n_splits=self.n_splits[0], n_repeats=self.repeats[0], random_state=self.random_state)
        total_folds = self.n_splits[0] * self.repeats[0]

        with tqdm(total=total_folds, disable=(not self.use_tqdm)) as t:
            for train_val_idx, test_idx in outer_cv.split(X, y):
                X_train_val, X_test = X[train_val_idx], X[test_idx]
                y_train_val, y_test = y[train_val_idx], y[test_idx]

                clf, best_model = self.tune_hyperparams(X_train_val, y_train_val)

                if self.tune_threshold:
                    best_threshold, best_model = tune_threshold_process(
                        best_model, X_train_val, y_train_val, 
                        splits = self.n_splits[1], repeats = self.repeats[1], 
                        random_state=self.random_state, 
                        scoring=self.tune_threshold_scoring)

                stats = self.store_info(best_model, X_test, y_test, train_val_idx, test_idx)

                t.set_postfix({val: stats[val] for val in self.tqdm_metrics})
                t.update()

        self.results_ = self.extra_info | {
            'models': self.models_, 'stats': self.stats_, 'cms': self.cms_, 'idxs': self.idxs_
        }
        return self # Return self for potential method chaining

# --- Example Usage ---
#
# # 1. Instantiate the runner with your model and parameters
# cv_runner = NestedCVRunner(
#     model=YourModel(),
#     param_grid={'param1': [1, 10], 'param2': [0.1, 0.5]}
# )
#
# # 2. Execute the process on your data
# cv_runner.run(X_data, y_data)
#
# # 3. Access the results
# trained_models = cv_runner.models_
# performance_stats = cv_runner.stats_
# final_results_dict = cv_runner.results_
