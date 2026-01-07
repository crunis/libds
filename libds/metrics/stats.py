from sklearn.metrics import confusion_matrix, roc_auc_score, \
  average_precision_score, brier_score_loss, log_loss
import numpy as np

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
        auc = roc_auc_score(y_test, y_scores, multi_class='ovr')
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