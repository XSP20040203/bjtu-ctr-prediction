"""Exercise independent group-mass correction and task decomposition."""
import unittest
import numpy as np
import pandas as pd
from scipy.special import logit
import export_best as e


class ExportChecks(unittest.TestCase):
    def test_independent_full_composition(self):
        raw = pd.DataFrame(dict(user_id=[1,1,2,2,3],task_id=[4,5,4,4,5],slot_id=[0]*5,spread_app_id=[1]*5))
        group = np.array([0,0,1,1,2])
        anchor = np.array([.01,.03,.02,.08,.015])
        current = anchor*np.array([.8,1.2,1.1,.9,1.05])
        predicted = np.array([.15,.035,.02])
        record = dict(formula='group_transfer',group='task',between=1.5,within=1.)
        daily = e.correct(anchor,predicted,group,'poisson',.125)
        values = dict(current=current,anchor=anchor,daily=daily)
        data = dict(frame=raw,values=values,logits={k:logit(v) for k,v in values.items()},
            groups={'task':e.group_codes(raw,['task_id'])})
        actual = e.compose(data,record)
        independent = e.independent_prediction(raw,current,anchor,predicted,group,record)
        np.testing.assert_allclose(actual,independent,rtol=0,atol=2e-16)

    def test_unchanged_group_mass_preserves_current(self):
        raw = pd.DataFrame({'task_id':[7,7,9,9]})
        anchor = np.array([.02,.03,.01,.04])
        current = np.array([.025,.028,.012,.035])
        groups = np.array([0,0,1,1])
        predicted = np.bincount(groups,weights=anchor)
        actual = e.independent_prediction(raw,current,anchor,predicted,groups,dict(between=1.5,within=1.))
        np.testing.assert_allclose(actual,current,rtol=0,atol=2e-16)


if __name__ == '__main__':
    unittest.main()
