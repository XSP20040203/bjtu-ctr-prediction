"""Content-adjusted source feedback using other users on the same source date."""
import numpy as np
import pandas as pd

READ = ['e_et', 'u_userId', 'label', 'cillabel', 'pro', 'i_docId', 'i_s_sourceId',
        'i_cat', 'e_ch', 'e_m', 'e_po', 'e_section', 'i_dtype', 'i_upTimes', 'i_dislikeTimes']
GROUPS = [['i_docId'], ['i_s_sourceId'], ['i_cat'], ['e_ch'], ['e_m'], ['e_po'],
          ['e_section'], ['i_dtype'], ['i_upTimes'], ['i_dislikeTimes'],
          ['i_cat', 'e_ch'], ['i_cat', 'e_po'], ['i_s_sourceId', 'e_ch'], ['i_docId', 'e_ch']]
ALPHAS = [5., 50.]


def leave_user_out(source, target, user_groups, fields, alpha):
    """Estimate item/context response without any outcomes from its querying user/day."""
    day_codes, days = pd.factorize(source.e_et, sort=True)
    group_codes, groups = pd.factorize(pd.MultiIndex.from_frame(source[['e_et']+fields]), sort=True)
    nusers = int(user_groups.max())+1
    pair = group_codes.astype('int64')*nusers+user_groups
    _, own = np.unique(pair, return_inverse=True)
    total_n = np.bincount(group_codes, minlength=len(groups)).astype('float64')
    own_n = np.bincount(own).astype('float64')
    day_n = np.bincount(day_codes, minlength=len(days)).astype('float64')
    user_n = np.bincount(user_groups, minlength=nusers).astype('float64')
    other_n = total_n[group_codes]-own_n[own]
    forecasts = []
    for i in range(target.shape[1]):
        y = target[:, i]
        total = np.bincount(group_codes, weights=y, minlength=len(groups))
        self_total = np.bincount(own, weights=y)
        day_total = np.bincount(day_codes, weights=y, minlength=len(days))
        user_total = np.bincount(user_groups, weights=y, minlength=nusers)
        prior = (day_total[day_codes]-user_total[user_groups]+.5)/(day_n[day_codes]-user_n[user_groups]+5.)
        estimate = (total[group_codes]-self_total[own]+alpha*prior)/(other_n+alpha)
        forecasts.append(estimate)
    return np.column_stack(forecasts), other_n


def build_quality(news, keys, cutoff):
    if set(news.columns) != set(READ):
        raise ValueError('Only declared separate-news fields are permitted')
    source = news.loc[news.e_et <= cutoff].copy()
    assert source.label.isin([-1, 1]).all() and np.all(keys[:, 0] <= cutoff)
    user_groups, user_index = pd.factorize(pd.MultiIndex.from_frame(source[['e_et', 'u_userId']]), sort=True)
    n = len(user_index)
    size = np.bincount(user_groups, minlength=n).astype('float64')
    clicked = source.label.eq(1).to_numpy(dtype='float64')
    read = np.clip(source.pro.to_numpy(dtype='float64'), 0, 100)/100*clicked
    liked = source.cillabel.eq(1).to_numpy(dtype='float64')
    target = np.column_stack([clicked, read, liked])
    observed = np.column_stack([np.bincount(user_groups, weights=target[:, i], minlength=n) for i in range(3)])
    arrays, names = [], []
    for fields in GROUPS:
        for alpha in ALPHAS:
            forecast, others = leave_user_out(source, target, user_groups, fields, alpha)
            expected = np.column_stack([np.bincount(user_groups, weights=forecast[:, i], minlength=n) for i in range(3)])
            variance = np.bincount(user_groups, weights=forecast[:, 0]*(1-forecast[:, 0]), minlength=n)
            probability = forecast[:, 0].clip(1e-6, 1-1e-6)
            surprise = np.bincount(user_groups, weights=-(clicked*np.log(probability)+(1-clicked)*np.log1p(-probability)), minlength=n)/size.clip(1)
            confidence = np.bincount(user_groups, weights=others/(others+alpha), minlength=n)/size.clip(1)
            coverage = np.bincount(user_groups, weights=others>0, minlength=n)/size.clip(1)
            log_others = np.bincount(user_groups, weights=np.log1p(others), minlength=n)/size.clip(1)
            block = np.column_stack([np.log1p(expected[:, 0]), expected[:, 0]/size.clip(1),
                        (observed[:, 0]-expected[:, 0])/np.sqrt(1+variance),
                        np.log((observed[:, 0]+.5)/(expected[:, 0]+.5)),
                        surprise, np.log1p(expected[:, 1]),
                        np.log((observed[:, 1]+.25)/(expected[:, 1]+.25)),
                        np.log1p(expected[:, 2]), np.log((observed[:, 2]+.5)/(expected[:, 2]+.5)),
                        confidence, coverage, log_others]).astype('float32')
            arrays.append(block)
            prefix = 'quality__'+'__'.join(fields)+f'__alpha{alpha:g}__'
            names += [prefix+s for s in ['log_expected_clicks', 'expected_ctr', 'click_pearson_residual',
                'log_click_observed_expected', 'source_response_surprisal', 'log_expected_read',
                'log_read_observed_expected', 'log_expected_likes', 'log_like_observed_expected',
                'other_user_confidence', 'other_user_coverage', 'mean_log_other_users_exposures']]
    values = np.column_stack(arrays)
    positions = user_index.get_indexer(pd.MultiIndex.from_arrays(keys.T))
    output = np.zeros((len(keys), values.shape[1]+1), dtype='float32')
    known = positions >= 0
    output[known, :-1] = values[positions[known]]
    output[:, -1] = ~known
    names += ['quality__news_missing']
    assert len(names) == output.shape[1] and len(set(names)) == len(names)
    assert np.isfinite(output).all()
    return output, names


def self_test():
    news = pd.DataFrame({name: [1, 1, 1, 1, 1, 1] for name in READ})
    news['e_et'] = [20191111, 20191111, 20191111, 20191111, 20191112, 20191112]
    news['u_userId'] = [1, 1, 2, 3, 1, 2]
    news['label'] = [1, -1, 1, -1, 1, -1]
    news['i_docId'] = ['a', 'b', 'a', 'a', 'a', 'b']
    keys = np.array([[20191111, 1], [20191111, 2], [20191111, 3], [20191111, 4]])
    before, names = build_quality(news, keys, 20191111)
    changed = news.copy()
    changed.loc[changed.e_et == 20191112, ['label', 'pro', 'i_cat']] = [-1, 99, 999]
    after, replay_names = build_quality(changed, keys, 20191111)
    np.testing.assert_array_equal(before, after)
    assert names == replay_names and before[-1, -1] == 1
    source = news.loc[news.e_et == 20191111].copy()
    group, _ = pd.factorize(pd.MultiIndex.from_frame(source[['e_et', 'u_userId']]), sort=True)
    target = source.label.eq(1).to_numpy(dtype='float64')[:, None]
    expectation, _ = leave_user_out(source, target, group, ['i_docId'], 5.)
    perturbed = target.copy()
    perturbed[group == 0] = 1-perturbed[group == 0]
    replay, _ = leave_user_out(source, perturbed, group, ['i_docId'], 5.)
    np.testing.assert_allclose(expectation[group == 0], replay[group == 0], rtol=0, atol=1e-15)
    return {'future_source_rows_excluded': True, 'query_user_outcomes_excluded_from_content_expectation': True,
            'unknown_user_day_explicit': True, 'feature_count': len(names)}


if __name__ == '__main__':
    print(self_test())
