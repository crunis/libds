from sklearn.datasets import make_classification
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from libds.estimators import GBCThreshold

import pytest

def test_can_be_instantiated():
    gbct = GBCThreshold()


def test_it_works():
    gbct = GBCThreshold()
    X, y = make_classification(
        n_samples=1000, n_features=20, n_informative=10,
        n_redundant=5, n_classes=2, random_state=42
        )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)
    
    gbct.fit(X_train, y_train)
    auc = roc_auc_score(y_test, gbct.predict_proba(X_test)[:,1])
    assert auc > 0.9


def test_threshold_works():
    gbct = GBCThreshold(random_state=42)
    X, y = make_classification(
        n_samples=1000, n_features=20, n_informative=10,
        n_redundant=5, n_classes=2, random_state=42,
        )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)
    
    gbct.fit(X_train, y_train)
    probs = gbct.predict_proba(X_test[:1,:])

    assert probs[0,0] == pytest.approx(0.41219117)
    assert probs[0,1] == pytest.approx(0.58780883)

    assert (gbct.classes_ == [0,1]).all()

    gbct.threshold = 0.2
    low_threshold_positives = gbct.predict(X_test).sum()
    gbct.threshold = 0.8
    high_threshold_positives = gbct.predict(X_test).sum()

    assert high_threshold_positives < low_threshold_positives
    