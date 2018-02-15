import os.path
import unittest

from dnnamo.tools.primops import PrimopsTool

from ..util import runtool, in_temporary_directory

TESTFILE = 'test/test_models/simple_nnet.py'

class TestPrimop(unittest.TestCase):
  def test_simply_run(self):
    testfile = os.path.abspath(TESTFILE)
    with in_temporary_directory():
      cmd='primops '+testfile
      runtool(PrimopsTool(), cmd)

  def test_run_dynamic(self):
    testfile = os.path.abspath(TESTFILE)
    with in_temporary_directory():
      cmd='primops '+testfile+' --timing'
      runtool(PrimopsTool(), cmd)
