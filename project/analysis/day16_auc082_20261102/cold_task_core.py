"""Identify unseen ad tasks and identity-specific inputs without reading outcomes."""
import numpy as np
import pandas as pd


def cold_mask(history,query,cutoff):
    if list(history)!=['pt_d','task_id'] or list(query)!=['pt_d','task_id']:
        raise ValueError('Only unlabeled date/task keys are allowed')
    if np.any(query.pt_d.to_numpy()!=cutoff):
        raise ValueError('Query dates must equal the prediction date')
    prior=history.loc[history.pt_d.lt(cutoff),'task_id'].unique()
    return ~np.isin(query.task_id.to_numpy(),prior)


def identity_columns(original_names):
    keep=[i for i,name in enumerate(original_names) if not name.startswith('rich_daily__quality__')]
    names=[original_names[i] for i in keep]+[f'quality_{i}' for i in range(337)]
    if len(names)!=1017:
        raise ValueError('Expected the verified1017-column full-quality row schema')
    blocked=['task_id','adv_id','hist_click_task','hist_close_task','ordered_click_history__ad_click_list_v001']
    indices=[i for i,name in enumerate(names) if any(part in name for part in blocked)]
    if not indices:
        raise ValueError('No task/creative-specific identity inputs found')
    return indices,[names[i] for i in indices]

