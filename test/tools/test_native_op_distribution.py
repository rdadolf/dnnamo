import unittest

from tools.native_op_distribution import Tool

from .util import runtool

class TestNativeOpDistribution(unittest.TestCase):
  def test_simple_run(self):
    testfile = 'test/examples/simple_nnet.py'
    cmd='native_op_distribution --framework=tf --noplot '+testfile
    runtool(Tool(), cmd)


  def test_multiple_models(self):
    testfiles = ['test/examples/simple_nnet.py', 'test/examples/simple_nnet.py']
    cmd = 'native_op_distribution --framework=tf --noplot '+' '.join(testfiles)

    runtool(Tool(), cmd)