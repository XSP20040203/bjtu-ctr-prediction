"""Chronological event probabilities for unordered history-change groups."""
from collections import Counter
import numpy as np
from scipy.special import expit, logit

CHANNELS = [('ad_click_list_v001', 'task_id'), ('ad_click_list_v002', 'adv_prim_id'),
            ('ad_click_list_v003', 'spread_app_id')]
FIELDS = ['pt_d', 'user_id', 'task_id', 'adv_prim_id', 'spread_app_id'] + [a for a, _ in CHANNELS]


def tokens(text):
    return frozenset(int(x) for x in str(text).split('^') if x not in ('', '-1', 'nan', 'None'))


def events(frame, independent=False):
    if list(frame) != FIELDS:
        raise ValueError('Only declared unlabeled date, user, candidate and history fields are allowed')
    dates, users = frame.pt_d.to_numpy(), frame.user_id.to_numpy()
    order = np.lexsort((users, dates))
    cuts = np.r_[0, 1 + np.flatnonzero((dates[order][1:] != dates[order][:-1]) |
                                     (users[order][1:] != users[order][:-1])), len(frame)]
    result = {}
    for field, target in CHANNELS:
        texts, candidates = frame[field].astype(str).to_numpy(), frame[target].to_numpy()
        cache, records = {}, {}
        for left, right in zip(cuts[:-1], cuts[1:]):
            rows = order[left:right]
            local_text = texts[rows]
            if len(set(local_text)) < 2:
                continue
            for text in set(local_text):
                if text not in cache:
                    cache[text] = tokens(text)
            states = [cache[text] for text in local_text]
            unique = set(states)
            if len(unique) < 2:
                continue
            local_candidates = candidates[rows]
            if independent:
                # Each original exposure independently enumerates its alternative states.
                for i in range(len(rows) - 1, -1, -1):
                    before, candidate = states[i], int(local_candidates[i])
                    if candidate in before:
                        continue
                    alternatives = [s for s in unique if candidate in s]
                    if not alternatives:
                        continue
                    member = np.array([j for j in range(len(rows))
                                       if states[j] == before and local_candidates[j] == candidate])
                    single = any(len(s - before) == 1 for s in alternatives)
                    clean = any(len(s - before) == 1 and not before - s for s in alternatives)
                    category = 1 + int(single) + int(single and len(member) == 1) + int(single and len(member) == 1 and clean)
                    key = (int(dates[rows[i]]), int(users[rows[i]]), candidate, tuple(sorted(before)))
                    records[key] = (np.sort(rows[member]), category)
            else:
                for before in unique:
                    positions = np.array([i for i, state in enumerate(states) if state == before])
                    counts = Counter(local_candidates[positions])
                    for candidate, size in counts.items():
                        if candidate in before:
                            continue
                        alternatives = [s for s in unique if candidate in s]
                        if not alternatives:
                            continue
                        single = any(len(s - before) == 1 for s in alternatives)
                        clean = any(len(s - before) == 1 and not before - s for s in alternatives)
                        category = 1 + int(single) + int(single and size == 1) + int(single and size == 1 and clean)
                        key = (int(dates[rows[0]]), int(users[rows[0]]), int(candidate), tuple(sorted(before)))
                        records[key] = (np.sort(rows[positions[local_candidates[positions] == candidate]]), category)
        ids = np.full(len(frame), -1, dtype='int32')
        group_dates, bins, sizes = [], [], []
        for index, key in enumerate(sorted(records)):
            rows, category = records[key]
            assert np.all(ids[rows] == -1)
            ids[rows] = index
            group_dates.append(key[0])
            bins.append(category)
            sizes.append(len(rows))
        result[field] = dict(ids=ids, dates=np.asarray(group_dates, dtype='int32'),
                             bins=np.asarray(bins, dtype='int8'), sizes=np.asarray(sizes, dtype='int32'))
    return result


def fit(grouped, dates, labels, cutoff, strength, recent=False, independent=False):
    dates, labels = np.asarray(dates), np.asarray(labels)
    if dates.shape != labels.shape or np.any(labels[dates >= cutoff] != -1):
        raise ValueError('Current and future labels must be masked')
    prior = dates < cutoff
    if not prior.any() or not np.isin(labels[prior], [0, 1]).all():
        raise ValueError('Observed binary earlier labels are required')
    if recent:
        prior &= dates == dates[prior].max()
    row_rate = (float(labels[prior].sum()) + .5) / (int(prior.sum()) + 1.)
    output = {}
    for channel, groups in grouped.items():
        ng = len(groups['dates'])
        chosen = groups['dates'] < cutoff
        if recent:
            chosen &= groups['dates'] == dates[prior].max()
        ids = groups['ids']
        eligible = (ids >= 0) & prior
        positives = np.zeros(ng, dtype='int32')
        if independent:
            for row in np.flatnonzero(eligible):
                positives[ids[row]] = max(positives[ids[row]], int(labels[row]))
        else:
            np.maximum.at(positives, ids[eligible], labels[eligible])
        counts, successes, rates = [], [], []
        for category in range(1, 5):
            cell = chosen & (groups['bins'] == category)
            count = int(cell.sum())
            success = int(positives[cell].sum())
            prior_event = float(np.mean(-np.expm1(groups['sizes'][cell] * np.log1p(-row_rate)))) if count else row_rate
            rates.append((success + strength * prior_event) / (count + strength))
            counts.append(count)
            successes.append(success)
        output[channel] = dict(counts=counts, successes=successes, rates=rates,
                               training_groups=int(chosen.sum()))
    return dict(channels=output, global_row_rate=row_rate, strength=float(strength), recent=bool(recent),
                max_training_date=int(dates[prior].max()), training_rows=int(prior.sum()))


def deltas(base, grouped, record, independent=False):
    from transition_posterior_core_v1 import prior_shift
    base = np.asarray(base, dtype='float64')
    if not np.all((base > 0) & (base < 1)):
        raise ValueError('Open-interval probabilities required')
    shifted = logit(base) + prior_shift(base, record['global_row_rate'])
    calibrated = expit(shifted)
    output = {}
    for channel, groups in grouped.items():
        ids = groups['ids']
        if len(ids) != len(base):
            raise ValueError('Aligned event groups required')
        known = ids >= 0
        ng = len(groups['dates'])
        log_survival = np.zeros(ng)
        if independent:
            for index in range(ng):
                log_survival[index] = np.log1p(-calibrated[ids == index]).sum()
        else:
            np.add.at(log_survival, ids[known], np.log1p(-calibrated[known]))
        probability_any = np.maximum(-np.expm1(log_survival), 1e-15)
        desired = np.asarray(record['channels'][channel]['rates'])[groups['bins'] - 1]
        value = np.zeros(len(base))
        # These are marginals of a mixture conditional on at least one group click.
        target = np.clip(calibrated[known] * desired[ids[known]] / probability_any[ids[known]], 1e-12, 1 - 1e-12)
        value[known] = np.maximum(logit(target) - shifted[known], 0.)
        output[channel] = value
    output['maximum'] = np.max(np.stack(list(output.values())), axis=0)
    return output


def predict(base, delta, weight, independent=False):
    base, delta = np.asarray(base), np.asarray(delta)
    if base.shape != delta.shape or not 0 <= weight <= 1 or not np.isfinite(delta).all():
        raise ValueError('Aligned finite corrections and a convex weight required')
    if independent:
        odds = base / (1 - base) * np.exp(weight * delta)
        return odds / (1 + odds)
    return expit(logit(base) + weight * delta)
