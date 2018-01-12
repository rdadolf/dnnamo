import unittest

from dnnamo.tools.native_op_distribution import NativeOpDistributionTool

from .util import runtool

class TestNativeOpDistribution(unittest.TestCase):
  def test_simple_run(self):
    testfile = 'test/test_models/simple_nnet.py'
    cmd='native_op_distribution --framework=tf --noplot '+testfile
    runtool(NativeOpDistributionTool(), cmd)


  def test_multiple_models(self):
    testfiles = ['test/test_models/simple_nnet.py', 'test/test_models/simple_nnet.py']
    cmd = 'native_op_distribution --framework=tf --noplot '+' '.join(testfiles)

    runtool(NativeOpDistributionTool(), cmd)
