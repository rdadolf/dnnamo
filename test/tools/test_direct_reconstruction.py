import unittest

from tools.direct_reconstruction import Tool

from .util import runtool

TESTFILE = 'test/test_models/simple_nnet.py'

class TestDirectReconstruction(unittest.TestCase):
  def test_simply_run(self):
    cmd='direct_reconstruction --noplot '+TESTFILE
    runtool(Tool(), cmd)
