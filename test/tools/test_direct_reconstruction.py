import unittest

from dnnamo.tools.direct_reconstruction import DirectReconstructionTool

from .util import runtool

TESTFILE = 'test/test_models/simple_nnet.py'

class TestDirectReconstruction(unittest.TestCase):
  def test_simply_run(self):
    cmd='direct_reconstruction --noplot '+TESTFILE
    runtool(DirectReconstructionTool(), cmd)
