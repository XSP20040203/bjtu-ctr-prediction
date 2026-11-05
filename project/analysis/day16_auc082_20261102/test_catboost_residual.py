import tempfile
from pathlib import Path
import unittest
import numpy as np
import pandas as pd
from catboost_userday_residual import CatBoost,Pool,save_model,load_model


class CatBoostTests(unittest.TestCase):
    def test_baseline_and_serialization(self):
        x=pd.DataFrame({"numeric":[0.,1.,2.,3.,4.,5.],"category":[1,1,2,2,3,3]})
        y=np.array([0,0,1,0,1,1])
        baseline=np.array([-2.,-1.,0.,1.,2.,3.])
        with_base=Pool(x,label=y,baseline=baseline,cat_features=["category"])
        model=CatBoost({"iterations":3,"depth":2,"loss_function":"Logloss","thread_count":1,
                        "allow_writing_files":False,"verbose":False,"random_seed":42})
        model.fit(with_base)
        pure=model.predict(Pool(x,cat_features=["category"]),prediction_type="RawFormulaVal")
        included=model.predict(with_base,prediction_type="RawFormulaVal")
        np.testing.assert_allclose(included,pure+baseline,rtol=0,atol=1e-12)
        with tempfile.TemporaryDirectory(prefix="cb_test_") as folder:
            path=Path(folder) / "模型.cbm"
            save_model(model,path)
            replay=load_model(path).predict(Pool(x,cat_features=["category"]),prediction_type="RawFormulaVal")
            np.testing.assert_array_equal(replay,pure)


if __name__=="__main__":
    unittest.main()
