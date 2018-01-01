import unittest

from tools.amdahl import Tool

from .util import runtool

MODELNAME = 'NativeOpSampleModel0'
TESTFILE = 'test/test_models/simple_nnet.py'
CACHEFILE = '/tmp/cachefile'

class TestNativeOpProfile(unittest.TestCase):
  def test_simple_run(self):
    for i in range(1,9):
      cmd='amdahl --framework=tf --noplot --threads '+str(i)+' '+TESTFILE
      runtool(Tool(), cmd)
