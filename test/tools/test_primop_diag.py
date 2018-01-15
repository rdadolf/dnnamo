import os.path
import unittest

from dnnamo.tools._primop_diag import PrimopDiagnosticTool

from ..util import runtool, in_temporary_directory

TESTFILE = 'test/test_models/simple_nnet.py'

class TestPrimopDiagnostic(unittest.TestCase):
  def test_simply_run(self):
    testfile = os.path.abspath(TESTFILE)
    with in_temporary_directory():
      cmd='_primop_diag '+testfile
      runtool(PrimopDiagnosticTool(), cmd)
