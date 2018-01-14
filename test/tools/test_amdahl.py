import os.path
import unittest

from dnnamo.tools.amdahl import AmdahlTool

from ..util import runtool, in_temporary_directory

MODELNAME = 'NativeOpSampleModel0'
TESTFILE = 'test/test_models/simple_nnet.py'
CACHEFILE = '/tmp/cachefile'

class TestNativeOpProfile(unittest.TestCase):
  def test_simple_run(self):
    testfile = os.path.abspath(TESTFILE)
    with in_temporary_directory():
      for i in range(1,9):
        cmd='amdahl --framework=tf --noplot --threads '+str(i)+' '+testfile
        runtool(AmdahlTool(), cmd)
