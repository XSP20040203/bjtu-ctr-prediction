"""Check unknown/padding semantics, row permutation, and independent encoders."""
import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from frozen_neural_embedding_core_v1 import CATEGORIES, SEQUENCES, FIELDS, encode, design

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'model_comparison'))


class FrozenEmbeddingTests(unittest.TestCase):
    def fixture(self):
        meta = dict(group_for_field={name: 'ids' for name in CATEGORIES + SEQUENCES},
                    vocabulary_sizes={'ids': 5})
        vocab = {'ids': {'-1': 2, '2': 3, '7': 4}}
        tables = {'ids': np.random.default_rng(15).normal(size=(5, 8)).astype('float32')}
        tables['ids'][0] = 0
        frame = pd.DataFrame({name: [2, 7, 111, -1] for name in CATEGORIES})
        frame['pt_d'] = 20191115
        for name in SEQUENCES:
            frame[name] = ['', '2^7^2.0^111^-1^7', '-1', 'nan']
        return frame[FIELDS], meta, vocab, tables

    def test_independent_encoders_and_features(self):
        frame, meta, vocab, tables = self.fixture()
        primary = encode(frame, meta, vocab, 20191115)
        other = encode(frame, meta, vocab, 20191115, True)
        for name in primary:
            np.testing.assert_array_equal(primary[name], other[name])
        np.testing.assert_array_equal(primary['user_id'], [3, 4, 1, 2])
        np.testing.assert_array_equal(primary[SEQUENCES[0]][1], [3, 4, 3, 1, 2])
        first = design(primary, tables, meta)
        second = design(other, tables, meta, True)
        np.testing.assert_array_equal(first, second)
        order = [2, 0, 3, 1]
        permuted = encode(frame.iloc[order], meta, vocab, 20191115)
        np.testing.assert_array_equal(first[order], design(permuted, tables, meta))
        np.testing.assert_array_equal(first[0, 48:96], np.zeros(48))

    def test_labels_and_future_metadata_rejected(self):
        frame, meta, vocab, _ = self.fixture()
        with self.assertRaises(ValueError):
            encode(frame.assign(label=0), meta, vocab, 20191115)
        frame.loc[0, 'pt_d'] = 20191116
        with self.assertRaises(ValueError):
            encode(frame, meta, vocab, 20191115)


if __name__ == '__main__':
    unittest.main()
