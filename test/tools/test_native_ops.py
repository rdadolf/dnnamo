import unittest

from tools.native_ops import Tool

from .util import runtool, cleanup_cachefile

TESTFILE = 'test/examples/simple_nnet.py'
SIMPLE_NNET_OP_COUNT = 89 # the number of static ops in the simple_nnet graph
OP_COUNT_THRESHOLD = 5 # eh, give it a little wiggle room
CACHEFILE = '/tmp/cachefile'

class TestNativeOps(unittest.TestCase):
  def test_simple_run(self):
    cmd='native_ops --framework=tf '+TESTFILE
    runtool(Tool(), cmd)

  def test_op_count(self):
    # NOTE: This is an extremely fragile test.
    #   The point is less to get a correct answer than it is to detect changes
    #   in the underlying libraries or inconsistencies in the way ops are being
    #   handled. If this test fails, check the output manually. If it looks sane,
    #   then reset the value here to whatever the current count is.

    cmd='native_ops --framework=tf '+TESTFILE
    t = Tool()
    runtool(t, cmd)
    opdiff = abs( len(t.data[0])-SIMPLE_NNET_OP_COUNT )
    assert opdiff<OP_COUNT_THRESHOLD, 'Native op count in simple_nnet doesnt match: '+str(len(t.data[0]))+' vs '+str(SIMPLE_NNET_OP_COUNT)+'(expected)'

  def test_cachefile(self):
    with cleanup_cachefile(CACHEFILE):
      cmd = 'native_ops --cachefile='+str(CACHEFILE)+' --writecache '+TESTFILE
      t = Tool()
      runtool(t, cmd)
      assert abs(len(t.data[0])-SIMPLE_NNET_OP_COUNT)<OP_COUNT_THRESHOLD, 'Native op count in simple_nnet doesnt match: '+str(len(t.data[0]))+' vs '+str(SIMPLE_NNET_OP_COUNT)+'(expected)'

      cmd2 = 'native_ops --cachefile='+str(CACHEFILE)+' --readcache '
      t = Tool()
      runtool(t, cmd2)
      assert abs(len(t.data[0])-SIMPLE_NNET_OP_COUNT)<OP_COUNT_THRESHOLD, 'Native op count in simple_nnet doesnt match: '+str(len(t.data[0]))+' vs '+str(SIMPLE_NNET_OP_COUNT)+'(expected)'

