import unittest
from typing import Any, Hashable
import hashlib

from . import run_exp_binary_iid_sequences
from . import run_exp_non_binary_iid
from . import run_exp_markov_chain
from . import run_exp_random_walk_with_memory
from . import run_exp_neighbourhood_size
from . import run_exp_random_ngram_model
from . import run_exp_suite


def canonicalize(x: Any) -> Hashable:
    if isinstance(x, Hashable):
        return x
    elif isinstance(x, dict):
        return tuple([(k, canonicalize(x[k])) for k in sorted(list(x.keys()))])
    elif isinstance(x, list):
        return tuple([canonicalize(y) for y in x])
    else:
        raise NotImplementedError()


class TestRunExp(unittest.TestCase):
    def test_binary_iid_sequences(self):
        run_exp_binary_iid_sequences.run_experiment(test_run=True)

    # @unittest.skip
    # def test_non_binary_iid(self):
    #     run_exp_non_binary_iid.run_experiment(test_run=True)

    def test_markov_chain(self):
        run_exp_markov_chain.run_experiment(test_run=True)

    def test_ngram_model(self):
        run_exp_random_walk_with_memory.run_experiment(test_run=True)

    def test_random_ngram_model(self):
        run_exp_random_ngram_model.run_experiment(test_run=True)

    # @unittest.skip
    # def test_neighbourhood_size(self):
    #     run_exp_neighbourhood_size.run_experiment(test_run=True)

    def test_the_suite(self):
        result = run_exp_suite.run_experiment(test_run=True)
        # known good hash of results - you can use this to make sure a refactor didn't affect results
        m = hashlib.sha256()
        m.update(str(canonicalize(result)).encode('ascii'))
        digest = m.hexdigest()
        self.assertEqual(digest[:8], "4e588b6c")
