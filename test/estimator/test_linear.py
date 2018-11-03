import numpy as np
import os
import unittest

from dnnamo.estimator import OLSEstimator

from ..util import in_temporary_directory

import pytest
class TestOLSEstimator(unittest.TestCase):
  #def setUp(self):
  # FIXME: generate test data if necessary

  def test_instantiate(self):
    est = OLSEstimator()

  def test_blind_predict(self):
    est = OLSEstimator()
    est.fit([[0]], [0])
    v = est.estimate([0])
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
    est.fit(line_x, line_y)

    v = est.estimate([100])
    assert np.abs(v-v_true)<10., 'Incorrect estimation'
    params = est.get_params()
    assert np.abs(m_true-params[0])<1., 'Incorrect slope fit'
    assert np.abs(b_true-params[1])<10., 'Incorrect intercept fit'
    v = est.estimate([100])
    assert np.abs(v-v_true)<10., 'Incorrect estimation'

    est2 = OLSEstimator()
    est2.set_params(params)
    v = est2.estimate([100])
    assert np.abs(v-v_true)<10., 'Incorrect estimation'

  def test_io(self):
    with in_temporary_directory() as d:
      filename = os.path.join(os.path.abspath(d), 'zero-linear.est')
      est = OLSEstimator()
      est.fit([[0,1,2,3]],[0])
      est.write(filename)
      # FIXME: flush?
      est2 = OLSEstimator()
      est2.read(filename)

      assert len(est.get_params())==len(est2.get_params())
      for i,(lhs,rhs) in enumerate(zip(est.get_params(), est2.get_params())):
        assert lhs==rhs, 'Parameter '+str(i)+' does not match: "'+str(lhs)+'" vs. "'+str(rhs)+'"'
