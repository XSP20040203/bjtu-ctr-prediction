"""Training-date-only inspection of unordered click-history snapshot changes."""
from collections import Counter
import numpy as np
import pandas as pd
from fit_stacker import ROOT, HERE, write


def tokens(value):
    return frozenset(int(x) for x in str(value).split('^') if x not in ['', '-1', 'nan', 'None'])


def main():
    destination = HERE/'unordered_snapshot_transition_prior14_v1.json'
    assert not destination.exists(), 'Preserve prior diagnostics'
    fields = ['pt_d','user_id','task_id','label','ad_click_list_v001']
    raw = pd.read_csv(ROOT/'2026ml/train.csv', usecols=fields,
                      dtype={'ad_click_list_v001': str})
    raw = raw[raw.pt_d == 20191114].reset_index(drop=True)
    flags = {key:np.zeros(len(raw),dtype=bool) for key in ['any_add','single_add','single_exposure','first_new_state']}
    state_users = 0
    for _, group in raw.groupby('user_id', sort=False):
        states = {name:tokens(name) for name in group.ad_click_list_v001.unique()}
        if len(states) < 2:
            continue
        state_users += 1
        for name, before in states.items():
            rows = group.index[group.ad_click_list_v001 == name].to_numpy()
            tasks = raw.task_id.to_numpy()[rows]
            counts = Counter(tasks)
            for after in states.values():
                added = after-before
                if not added:
                    continue
                match = np.isin(tasks,list(added))
                flags['any_add'][rows[match]] = True
                if len(added) != 1:
                    continue
                flags['single_add'][rows[match]] = True
                eligible = rows[match & np.array([counts[value] == 1 for value in tasks])]
                flags['single_exposure'][eligible] = True
                if before.issubset(after):
                    flags['first_new_state'][eligible] = True
    labels = raw.label.to_numpy()
    report = dict(date=20191114, scope='Only historical training date14 evaluated; no labels from dates15/16 used in these statistics; no date17 data loaded',
                  state_change_users=state_users, total_rows=len(raw), total_clicks=int(labels.sum()),groups={})
    for name, mask in flags.items():
        report['groups'][name] = dict(rows=int(mask.sum()), clicks=int(labels[mask].sum()),
                                      click_rate=float(labels[mask].mean()) if mask.any() else None)
    write(destination,report)
    print(report,flush=True)


if __name__ == '__main__':
    main()
