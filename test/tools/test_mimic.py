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

      #FIXME: add actual test here
