import numpy as np
import unittest

from dnnamo.estimator import OLSEstimator

import pytest
class TestOLSEstimator(unittest.TestCase):
  #def setUp(self):
  # FIXME: generate test data if necessary

  def test_instantiate(self):
    est = OLSEstimator()

  def test_blind_predict(self):
    est = OLSEstimator()
    est.fit('zero', [[0]], [0])
    v = est.estimate('zero', [0])
    assert (v+1), 'Estimate returned non-numeric value: '+str(v)

  def test_reproducible_parameters(self):
    N = 1000
    np.random.seed(13)
    noise = 20*np.random.randn(N)
    line_x = np.random.random_integers(1,1000,(N,1))
    m_true,b_true = 3,20
    line_y = m_true*line_x[:,0] + b_true + noise
    v_true = m_true*100 + b_true

    est = OLSEstimator()
    est.fit('zero', line_x, line_y)

    v = est.estimate('zero', [100])
    assert np.abs(v-v_true)<10., 'Incorrect estimation'
    params = est.get_params('zero')
    assert np.abs(m_true-params[0])<1., 'Incorrect slope fit'
    assert np.abs(b_true-params[1])<10., 'Incorrect intercept fit'
    v = est.estimate('zero', [100])
    assert np.abs(v-v_true)<10., 'Incorrect estimation'

    est2 = OLSEstimator()
    est2.set_params('zero',params)
    v = est2.estimate('zero', [100])
    assert np.abs(v-v_true)<10., 'Incorrect estimation'

