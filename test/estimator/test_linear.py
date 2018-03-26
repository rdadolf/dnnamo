import unittest

from dnnamo.estimator import OLSEstimator

import pytest
@pytest.mark.xfail()
class TestOLSEstimator(unittest.TestCase):
  def setUp(self):
    pass # FIXME: generate test data

  def test_instantiate(self):
    est = OLSEstimator()
