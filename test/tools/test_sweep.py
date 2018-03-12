import unittest

from dnnamo.tools.sweep import SweepTool

from ..util import runtool, in_temporary_directory

class TestSweep(unittest.TestCase):
  def test_simply_run(self):
    with in_temporary_directory():
      runtool(SweepTool(), 'sweep Primop_zerp')
