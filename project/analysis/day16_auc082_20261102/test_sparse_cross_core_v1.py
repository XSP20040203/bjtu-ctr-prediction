import unittest
import numpy as np
import pandas as pd
from sparse_cross_core_v1 import FIELDS,CATEGORIES,CHANNELS,encode,encode_chunks,margin,hashes,scalar_hash


class SparseTests(unittest.TestCase):
    def data(self):
        d={name:[-1,2,3,2] for name in CATEGORIES}
        d.update({name:['1^2^2','3','-1','4^5'] for name in CHANNELS})
        return pd.DataFrame(d)

    def test_hash_independent_and_encoding(self):
        a=np.array([-1,0,2,987654321],dtype='int64')
        b=np.array([6,7,8,9],dtype='int64')
        np.testing.assert_array_equal(hashes('sample',a,b),np.array([scalar_hash('sample',x,y) for x,y in zip(a,b)],dtype='uint64'))
        d=self.data();x=encode(d,1024);z=encode(d,1024,True)
        self.assertEqual((x!=z).nnz,0)
        self.assertEqual((x!=encode_chunks(d,1024,chunk_size=1)).nnz,0)
        d['label']=[0,1,1,0]
        self.assertEqual((x!=encode(d,1024)).nnz,0)
        permutation=[3,1,0,2]
        self.assertEqual((x!=encode(d.iloc[permutation],1024)[np.argsort(permutation)]).nnz,0)

    def test_sparse_margin_replay(self):
        x=encode(self.data(),1024)
        coefficients=np.random.default_rng(42).normal(size=1024)
        a=margin(x,coefficients,.3)
        b=margin(x,coefficients,.3,True)
        np.testing.assert_allclose(a,b,rtol=0,atol=1e-12)
        with self.assertRaises(ValueError):encode(self.data(),1000)


if __name__=='__main__':unittest.main()
