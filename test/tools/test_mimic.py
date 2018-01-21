import os.path
import unittest

from dnnamo.tools._mimic import MimicTool

from ..util import runtool, in_temporary_directory

TESTFILE = 'test/test_models/simple_nnet.py'

class TestMimic(unittest.TestCase):
  def test_simply_run(self):
    testfile = os.path.abspath(TESTFILE)
    with in_temporary_directory():
      cmd='_mimic '+testfile
      runtool(MimicTool(), cmd)

  def test_profile_detail(self):
    testfile = os.path.abspath(TESTFILE)
    with in_temporary_directory():
      cmd='_mimic --detail profile '+testfile
      tool = MimicTool()
      runtool(tool, cmd)

      true_time, mimic_time, components = tool.data[testfile]
      print 'true_time:',true_time
      print 'mimic_time:',mimic_time

      assert true_time>0, 'Invalid true time'
      assert mimic_time>0, 'Invalid mimic time'

      assert true_time > mimic_time, 'Impossible mimic time'

      # This test is not a true constraint, but if it is not true, then there
      # may be a problem.
      assert true_time < 3*mimic_time, 'Mimic time substantially different than true_time. This could be an artifact of testing, but it may indicate a problem.'

