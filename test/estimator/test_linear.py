import unittest

from dnnamo.estimator import OLSEstimator

import pytest
class TestOLSEstimator(unittest.TestCase):
  def setUp(self):
    pass # FIXME: generate test data

  def test_instantiate(self):
    est = OLSEstimator()

  def test_blind_predict(self):
    est = OLSEstimator()
    est.fit('zero', [[0]], [0])
    v = est.estimate('zero', [0])
    assert (v+1), 'Estimate returned non-numeric value: '+str(v)

  def test_reproducible_parameters(self):
    pass
    # FIXME:

  # FIXME: more!
