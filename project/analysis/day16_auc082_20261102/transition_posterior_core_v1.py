"""Unordered same-day task-history evidence and strictly prior smoothed rates."""
from collections import Counter
import numpy as np
from scipy.special import expit, logit

FIELDS = ['pt_d','user_id','task_id','ad_click_list_v001']


def task_bins(frame, independent=False):
    if list(frame) != FIELDS:
        raise ValueError('Only declared unlabeled fields are accepted')
    output = np.zeros(len(frame),dtype='int8')
    dates, users, tasks = (frame[n].to_numpy() for n in FIELDS[:3])
    strings = frame.ad_click_list_v001.astype(str).to_numpy()
    order = np.lexsort((users,dates))
    cuts = np.r_[0,1+np.flatnonzero((dates[order][1:] != dates[order][:-1]) |
                                  (users[order][1:] != users[order][:-1])),len(frame)]
    cache = {}
    for left,right in zip(cuts[:-1],cuts[1:]):
        rows = order[left:right]
        texts = strings[rows]
        if len(set(texts)) < 2:
            continue
        for text in set(texts):
            if text not in cache:
                cache[text] = frozenset(int(t) for t in text.split('^')
                    if t not in ['', '-1', 'nan', 'None'])
        states = [cache[t] for t in texts]
        unique = set(states)
        if len(unique) < 2:
            continue
        candidates = tasks[rows]
        if independent:
            for i,(before,candidate) in enumerate(zip(states,candidates)):
                if candidate in before:
                    continue
                additions = [(len(after-before),len(before-after)) for after in unique if candidate in after]
                if not additions:
                    continue
                value = 1
                if any(a == 1 for a,r in additions):
                    value = 2
                    count = sum(s == before and c == candidate for s,c in zip(states,candidates))
                    if count == 1:
                        value = 3
                        if any(a == 1 and r == 0 for a,r in additions):
                            value = 4
                output[rows[i]] = value
        else:
            for before in unique:
                positions = np.array([i for i,state in enumerate(states) if state == before])
                counts = Counter(candidates[positions])
                for candidate,count in counts.items():
                    if candidate in before:
                        continue
                    additions = [(len(after-before),len(before-after)) for after in unique if candidate in after]
                    if not additions:
                        continue
                    value = 1
                    if any(a == 1 for a,r in additions):
                        value = 2
                        if count == 1:
                            value = 3
                            if any(a == 1 and r == 0 for a,r in additions):
                                value = 4
                    output[rows[positions[candidates[positions] == candidate]]] = value
    return output


def fit_rates(bins, dates, labels, cutoff, strength, recent, independent=False):
    bins, dates, labels = map(np.asarray,(bins,dates,labels))
    if not (bins.shape == dates.shape == labels.shape) or np.any(labels[dates >= cutoff] != -1):
        raise ValueError('Aligned arrays with hidden current/future targets are required')
    prior = dates < cutoff
    if not prior.any() or not np.isin(labels[prior],[0,1]).all():
        raise ValueError('Binary prior labels are required')
    if recent:
        prior &= dates == dates[prior].max()
    global_rate = (float(labels[prior].sum())+.5)/(int(prior.sum())+1.)
    if independent:
        counts = np.array([np.count_nonzero(prior & (bins == b)) for b in range(5)])
        clicks = np.array([labels[prior & (bins == b)].sum() for b in range(5)],dtype='float64')
    else:
        counts = np.bincount(bins[prior],minlength=5)
        clicks = np.bincount(bins[prior],weights=labels[prior],minlength=5)
    rates = (clicks+strength*global_rate)/(counts+strength)
    return dict(counts=counts.tolist(),clicks=clicks.tolist(),rates=rates.tolist(),
        global_rate=global_rate,max_training_date=int(dates[prior].max()),
        training_rows=int(prior.sum()),strength=strength,recent=recent)


def prior_shift(base, target_rate):
    z = logit(np.asarray(base))
    lo,hi = -40.,40.
    for _ in range(60):
        middle = (lo+hi)/2
        if expit(z+middle).mean() < target_rate:
            lo = middle
        else:
            hi = middle
    return (lo+hi)/2


def predict(base,bins,record,threshold,weight,independent=False,shift=None):
    base,bins = np.asarray(base),np.asarray(bins)
    if shift is None:
        shift = prior_shift(base,record['global_rate'])
    target = logit(np.asarray(record['rates']))[bins]-shift
    # Evidence is used as a lower bound; all other original rows are unchanged.
    z = logit(base)
    delta = weight*np.maximum(target-z,0.)
    if independent:
        odds = base/(1-base)*np.exp(delta)
        p = odds/(1+odds)
    else:
        p = expit(z+delta)
    return np.where(bins >= threshold,p,base)
