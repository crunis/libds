# V1.4.3 16/10/2025
# New function names
# change FN% and TP% div by 0 to Nan
# add get_stats2

import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score, average_precision_score

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


def get_stats_from_ypred(y_test, y_pred, y_scores=None, digits=None, add_percentage_sign=False):
    cm = confusion_matrix(y_test, y_pred)
    stats = get_stats_from_cm(cm, digits=digits, add_percentage_sign=add_percentage_sign)

    if y_scores is not None:
        auc = roc_auc_score(y_test, y_scores)
        auc_pr = average_precision_score(y_test, y_scores)
        stats |= {'auc': auc, 'auc-pr': auc_pr}

    return stats