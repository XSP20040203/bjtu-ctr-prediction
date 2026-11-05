"""Label-free state-change features within a supplied user/date advertisement batch."""
from collections import Counter
import numpy as np

CHANNELS = [(f'ad_{action}_list_v00{version}', target)
            for action in ['click', 'close']
            for version, target in [(1, 'task_id'), (2, 'adv_prim_id'), (3, 'spread_app_id')]]
FIELDS = ['pt_d', 'user_id', 'task_id', 'adv_prim_id', 'spread_app_id'] + [field for field, _ in CHANNELS]
METRICS = ['log_states', 'log_state_rows', 'log_candidate_state_rows', 'state_exposure_share',
           'candidate_absent_added_elsewhere', 'log_min_added_tokens', 'log_min_removed_tokens',
           'single_added_token', 'single_added_unique_exposure', 'single_added_without_removal',
           'inverse_matching_exposures', 'log_matching_other_states']
NAMES = [f'unordered__{field}__{metric}' for field, _ in CHANNELS for metric in METRICS]


def tokenize(value):
    return frozenset(int(x) for x in str(value).split('^') if x not in ['', '-1', 'nan', 'None'])


def build(data, independent=False):
    """Only the supplied input columns are read; rows and dates are permutation invariant.

    A distinct state is a set of history tokens. A token can be newly present in
    another state, but no chronological order is inferred from row IDs. Features
    are zero when the user/date has only one state in that history channel.
    """
    data = data[FIELDS].reset_index(drop=True)
    size = len(data)
    output = np.zeros((size, len(NAMES)), dtype='float32')
    if not size:
        return output, list(NAMES)
    dates = data.pt_d.to_numpy()
    users = data.user_id.to_numpy()
    order = np.lexsort((users, dates))
    boundaries = np.r_[0, np.flatnonzero((dates[order][1:] != dates[order][:-1]) |
                                        (users[order][1:] != users[order][:-1])) + 1, size]
    token_cache = {}
    for channel, (field, target) in enumerate(CHANNELS):
        texts = data[field].astype(str).to_numpy()
        targets = data[target].to_numpy()
        for left, right in zip(boundaries[:-1], boundaries[1:]):
            rows = order[left:right]
            raw_states = texts[rows]
            if np.all(raw_states == raw_states[0]):
                continue
            for text in np.unique(raw_states):
                if text not in token_cache:
                    token_cache[text] = tokenize(text)
            states = sorted(set(token_cache[text] for text in raw_states), key=lambda s: tuple(sorted(s)))
            if len(states) <= 1:
                continue
            row_states = [token_cache[text] for text in raw_states]
            row_targets = targets[rows]
            block = np.zeros((len(rows), len(METRICS)), dtype='float32')
            if independent:
                # Direct per-exposure enumeration independently checks pooled state statistics.
                for i, (state, candidate) in enumerate(zip(row_states, row_targets)):
                    same = np.array([other == state for other in row_states])
                    count = int(np.sum(same & (row_targets == candidate)))
                    block[i, :4] = [np.log1p(len(states)), np.log1p(same.sum()),
                                     np.log1p(count), count / same.sum()]
                    matches = [other for other in states if candidate not in state and candidate in other]
                    if not matches:
                        continue
                    adds = [len(other-state) for other in matches]
                    removes = [len(state-other) for other in matches]
                    single = any(number == 1 for number in adds)
                    clean = any(a == 1 and r == 0 for a, r in zip(adds, removes))
                    block[i, 4:] = [1, np.log1p(min(adds)), np.log1p(min(removes)),
                                     single, single and count == 1, clean,
                                     1 / max(count, 1), np.log1p(len(matches))]
            else:
                for state in states:
                    positions = np.array([i for i, other in enumerate(row_states) if other == state])
                    counts = Counter(row_targets[positions])
                    for candidate, count in counts.items():
                        local = positions[row_targets[positions] == candidate]
                        block[local, :4] = [np.log1p(len(states)), np.log1p(len(positions)),
                                           np.log1p(count), count / len(positions)]
                        if candidate in state:
                            continue
                        matches = [other for other in states if candidate in other]
                        if not matches:
                            continue
                        adds = [len(other-state) for other in matches]
                        removes = [len(state-other) for other in matches]
                        single = any(number == 1 for number in adds)
                        clean = any(a == 1 and r == 0 for a, r in zip(adds, removes))
                        block[local, 4:] = [1, np.log1p(min(adds)), np.log1p(min(removes)),
                                            single, single and count == 1, clean,
                                            1 / max(count, 1), np.log1p(len(matches))]
            start = channel * len(METRICS)
            output[rows, start:start+len(METRICS)] = block
    if not np.isfinite(output).all():
        raise ValueError('Nonfinite unordered snapshot features')
    return output, list(NAMES)
