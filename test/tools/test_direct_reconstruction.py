import os.path
import unittest

from dnnamo.tools.direct_reconstruction import DirectReconstructionTool

from ..util import runtool, in_temporary_directory

TESTFILE = 'test/test_models/simple_nnet.py'

class TestDirectReconstruction(unittest.TestCase):
  def test_simply_run(self):
    testfile = os.path.abspath(TESTFILE)
    with in_temporary_directory():
      cmd='direct_reconstruction --noplot '+testfile
      runtool(DirectReconstructionTool(), cmd)
