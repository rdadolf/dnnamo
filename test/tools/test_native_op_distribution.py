import unittest
import os.path

from dnnamo.tools.native_op_distribution import NativeOpDistributionTool

from ..util import runtool, in_temporary_directory

class TestNativeOpDistribution(unittest.TestCase):
  def test_simple_run(self):
    testfile = os.path.abspath('test/test_models/simple_nnet.py')
    with in_temporary_directory():
      cmd='native_op_distribution --framework=tf --noplot '+testfile
      runtool(NativeOpDistributionTool(), cmd)

  def test_multiple_models(self):
    testfiles = map(os.path.abspath, ['test/test_models/simple_nnet.py', 'test/test_models/simple_nnet.py'])
    with in_temporary_directory():
      cmd = 'native_op_distribution --framework=tf --noplot '+' '.join(testfiles)

      runtool(NativeOpDistributionTool(), cmd)
