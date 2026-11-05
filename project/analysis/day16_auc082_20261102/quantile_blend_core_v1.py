"""Label-free score-scale matching and convex probability combinations."""
import numpy as np
from scipy.special import expit, logit
from scipy.stats import rankdata


def matched_logit(base, margin, independent=False):
    base = np.asarray(base, dtype='float64')
    margin = np.asarray(margin, dtype='float64')
    if base.ndim != 1 or margin.shape != base.shape or not len(base):
        raise ValueError('Nonempty aligned score vectors are required')
    if not np.isfinite(margin).all() or not np.all((base > 0) & (base < 1)):
        raise ValueError('Finite margins and open-interval probabilities are required')
    n = len(base)
    if independent:
        _, inverse, counts = np.unique(margin, return_inverse=True, return_counts=True)
        ranks = (np.cumsum(counts)-counts/2)[inverse]-.5
        reference = np.sort(np.log(base)-np.log1p(-base))
        lower = np.floor(ranks).astype('int64')
        upper = np.ceil(ranks).astype('int64')
        fraction = ranks-lower
        return reference[lower]*(1-fraction)+reference[upper]*fraction
    ranks = rankdata(margin, method='average')-1.
    return np.interp(ranks, np.arange(n), np.sort(logit(base)))


def blend(base, target_logit, mask, weight, independent=False):
    base = np.asarray(base, dtype='float64')
    target_logit = np.asarray(target_logit, dtype='float64')
    mask = np.asarray(mask, dtype=bool)
    if target_logit.shape != base.shape or mask.shape != base.shape or not 0 <= weight <= 1:
        raise ValueError('Aligned vectors and a convex weight are required')
    if weight == 0:
        return base.copy()
    z = logit(base)
    if independent:
        odds = base/(1-base)*np.exp(weight*(target_logit-z))
        adjusted = odds/(1+odds)
    else:
        adjusted = expit((1-weight)*z+weight*target_logit)
    return np.where(mask, adjusted, base)
