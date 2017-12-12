import unittest
import nnmodel
from synth import DGraphTestCase

class TestSolver(DGraphTestCase):
  @unittest.SkipTest
  def test_valid_eval_variables(self):
    VALID_VARS = ['time']

    for var in VALID_VARS:
      dg = self.synth_dgraph()
      solver = nnmodel.Solver(dg, devicemap={'test'})
      value = solver.eval(var)
      assert value>=0, 'Invalid value produced by eval: '+str(value)
