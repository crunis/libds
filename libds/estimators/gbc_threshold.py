import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted, check_array, check_X_y
from sklearn.utils.class_weight import compute_sample_weight

#############################
## Wraps GradientBoostingClassifier to implement threshold

class GBCThreshold(ClassifierMixin, BaseEstimator):
    OWN_PARAMETERS="threshold use_weights".split()
    
    def __init__(self, threshold=None, use_weights=False, **kwargs):
        self.threshold = threshold
        self.use_weights = use_weights
        self._gbc = GradientBoostingClassifier(**kwargs)


    def fit(self, X, y, **kwargs):
        X, y = check_X_y(X, y, accept_sparse=False)
        self.classes_ = np.unique(y)
        
        if self.use_weights:
            sample_weight = compute_sample_weight(class_weight='balanced', y=y)
            self._gbc.fit(X, y, **(kwargs | dict(sample_weight=sample_weight)))
        else:
            self._gbc.fit(X, y, **kwargs)

        return self


    def predict(self, X):
        check_is_fitted(self)
        check_array(X, accept_sparse=False)
        
        if self.threshold is None:
            y_pred = self._gbc.predict(X)
        else:
            y_pred = (self._gbc.predict_proba(X)[:,1]>self.threshold).astype(int)

        return y_pred


    def __getattr__(self, name):
        if name in self.OWN_PARAMETERS:
            return self.__getattribute__(name)
        
        return self._gbc.__getattribute__(name)
    
    
    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            if parameter in self.OWN_PARAMETERS:
                setattr(self, parameter, value)
            else:
                setattr(self._gbc, parameter, value)

        return self
    
    
    def get_params(self, deep=True):
        # We want to appear like a single estimator, so we integrate GBC params into us
        # and ignore deep
        params = super().get_params(deep=deep)
        gbc_params = self._gbc.get_params(deep=True)
        params.update(gbc_params)
        
        return params