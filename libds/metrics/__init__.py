from sklearn.metrics import confusion_matrix
from pycaret.classification import remove_metric, add_metric


def get_items(y, y_pred):
    cm = confusion_matrix(y, y_pred)
    tn, fp, fn, tp = [item for row in cm for item in row]
    n0  = tn + fn
    n1  = fp + tp
    return tn, fp, fn, tp, n0, n1

def pos0(y, y_pred, **kwargs):
    tn, fp, fn, tp, n0, n1 = get_items(y, y_pred)
    return fn

def pos1(y, y_pred, **kwargs):
    tn, fp, fn, tp, n0, n1 = get_items(y, y_pred)
    return tp

def n(y, y_pred, **kwargs):
    return len(y)

def n0(y, y_pred, **kwargs):
    tn, fp, fn, tp, n0, n1 = get_items(y, y_pred)
    return n0

def n1(y, y_pred, **kwargs):
    tn, fp, fn, tp, n0, n1 = get_items(y, y_pred)
    return n1

def n0p(y, y_pred, **kwargs):
    tn, fp, fn, tp, n0, n1 = get_items(y, y_pred)
    return round(n0*100 / (n0+n1),2)

def n1p(y, y_pred, **kwargs):
    tn, fp, fn, tp, n0, n1 = get_items(y, y_pred)
    return round(n1*100 / (n0+n1),2)

def pos0(y, y_pred, **kwargs):
    tn, fp, fn, tp, n0, n1 = get_items(y, y_pred)
    return fn

def pos1(y, y_pred, **kwargs):
    tn, fp, fn, tp, n0, n1 = get_items(y, y_pred)
    return tp

def pos0p(y, y_pred, **kwargs):
    tn, fp, fn, tp, n0, n1 = get_items(y, y_pred)
    return round(fn*100/n0,2)

def pos1p(y, y_pred, **kwargs):
    tn, fp, fn, tp, n0, n1 = get_items(y, y_pred)
    return round(tp*100/n1,2)

def sens(y, y_pred, **kwargs):
    tn, fp, fn, tp, n0, n1 = get_items(y, y_pred)
    return (tp/(tp+fn))

def spec(y, y_pred, **kwargs):
    tn, fp, fn, tp, n0, n1 = get_items(y, y_pred)
    return tn / (tn+fp)

def npv(y, y_pred, **kwargs):
    tn, fp, fn, tp, n0, n1 = get_items(y, y_pred)
    return 100* tn / (fn+tn)

def ppv(y, y_pred, **kwargs):
    tn, fp, fn, tp, n0, n1 = get_items(y, y_pred)
    return 100*tp / (fp+tp)

METRICS = [
    ('n', n),
    ('n0', n0),
    ('n1', n1),
    ('n0%', n0p),
    ('n1%', n1p),
    ('pos0', pos0),
    ('pos1', pos1),
    ('pos0%', pos0p),
    ('pos1%', pos1p),
    ('sens', sens),
    ('spec', spec),
    ('npv', npv),
    ('ppv', ppv)
]


def force_add_metric(metric_name, metric_func, verbose=False):
    try:
        remove_metric(metric_name)
    except:
        if verbose:
            print(f"couldnt remove {metric_name}")
    if verbose:
        print(f"Adding metric {metric_name}...")
    add_metric(metric_name, metric_name, metric_func, target = 'pred')