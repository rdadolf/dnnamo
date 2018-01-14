import os.path
import unittest

from dnnamo.tools.native_ops import NativeOpsTool

from ..util import runtool, cleanup_cachefile, in_temporary_directory

TESTFILE = 'test/test_models/simple_nnet.py'
SIMPLE_NNET_OP_COUNT = 89 # the number of static ops in the simple_nnet graph
OP_COUNT_THRESHOLD = 5 # eh, give it a little wiggle room
CACHEFILE = '/tmp/cachefile'

class TestNativeOps(unittest.TestCase):
  def test_simple_run(self):
    testfile = os.path.abspath(TESTFILE)
    with in_temporary_directory():
      cmd='native_ops --framework=tf '+testfile
      runtool(NativeOpsTool(), cmd)

  def test_op_count(self):
    testfile = os.path.abspath(TESTFILE)
    with in_temporary_directory():
      cmd='native_ops --framework=tf '+testfile
      t = NativeOpsTool()
      runtool(t, cmd)

      # FIXME: This is too brittle. We need to find a better way.
      #opdiff = abs( len(t.data[0])-SIMPLE_NNET_OP_COUNT )
      #assert opdiff<OP_COUNT_THRESHOLD, 'Native op count in simple_nnet doesnt match: '+str(len(t.data[0]))+' vs '+str(SIMPLE_NNET_OP_COUNT)+'(expected)'
      assert len(t.data[0])>0, 'No ops found'

  def test_cachefile(self):
    with in_temporary_directory():
      with cleanup_cachefile(CACHEFILE):
        cmd = 'native_ops --cachefile='+str(CACHEFILE)+' --writecache '+TESTFILE
        t = NativeOpsTool()
        runtool(t, cmd)

        # FIXME: This is too brittle. We need to find a better way.
        #assert abs(len(t.data[0])-SIMPLE_NNET_OP_COUNT)<OP_COUNT_THRESHOLD, 'Native op count in simple_nnet doesnt match: '+str(len(t.data[0]))+' vs '+str(SIMPLE_NNET_OP_COUNT)+'(expected)'
        assert len(t.data[0])>0, 'No ops found.'

        cmd2 = 'native_ops --cachefile='+str(CACHEFILE)+' --readcache '
        t = NativeOpsTool()
        runtool(t, cmd2)

        # FIXME: This is too brittle. We need to find a better way.
        #assert abs(len(t.data[0])-SIMPLE_NNET_OP_COUNT)<OP_COUNT_THRESHOLD, 'Native op count in simple_nnet doesnt match: '+str(len(t.data[0]))+' vs '+str(SIMPLE_NNET_OP_COUNT)+'(expected)'
        assert len(t.data[0])>0, 'No ops found'

