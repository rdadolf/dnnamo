import unittest
import dnnamo
from synth import AbstractGraphTestCase

class TestSolver(AbstractGraphTestCase):
  @unittest.SkipTest
  def test_valid_eval_variables(self):
    VALID_VARS = ['time']

    for var in VALID_VARS:
      g = self.synth_absgraph()
      solver = dnnamo.Solver(g, devicemap={'test'})
      value = solver.eval(var)
      assert value>=0, 'Invalid value produced by eval: '+str(value)
