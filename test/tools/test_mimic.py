import os.path
import unittest

from dnnamo.tools.mimic import MimicTool

from ..util import runtool, in_temporary_directory

TESTFILE = 'test/test_models/simple_nnet.py'

class TestMimic(unittest.TestCase):
  def test_simply_run(self):
    testfile = os.path.abspath(TESTFILE)
    with in_temporary_directory():
      cmd='mimic '+testfile
      runtool(MimicTool(), cmd)

  def test_profile_detail(self):
    testfile = os.path.abspath(TESTFILE)
    with in_temporary_directory():
      cmd='mimic --detail profile '+testfile
      tool = MimicTool()
      runtool(tool, cmd)

      wall_time, true_time = tool.data['wall_time'],tool.data['true_time']
      print 'wall_time:',wall_time
      print 'true_time:',true_time

      assert wall_time>0, 'Invalid wall time'
      assert true_time>0, 'Invalid true time'

      assert wall_time > true_time, 'Impossible true time'
