import unittest

from dnnamo.tools.native_op_profile import NativeOpProfileTool

from .util import runtool, cleanup_cachefile

MODELNAME = 'NativeOpSampleModel0'
TESTFILE = 'test/test_models/simple_nnet.py'
MODELNAMES = ['NativeOpSampleModel0', 'NativeOpSampleModel0']
TESTFILES = ['test/test_models/simple_nnet.py', 'test/test_models/simple_nnet.py']
SIMPLE_NNET_OP_COUNT = 27
OP_COUNT_THRESHOLD = 3
CACHEFILE = '/tmp/cachefile'

class TestNativeOpProfile(unittest.TestCase):
  def test_simple_run(self):
    cmd='native_op_profile --framework=tf --noplot '+TESTFILE
    runtool(NativeOpProfileTool(), cmd)

  def test_multiple_models(self):
    cmd = 'native_op_profile --framework=tf --noplot '+' '.join(TESTFILES)
    runtool(NativeOpProfileTool(), cmd)

  def test_zero_threshold(self):
    cmd='native_op_profile --framework=tf --noplot --threshold 0 --print '+TESTFILE
    t = NativeOpProfileTool()
    runtool(t, cmd)
    print t.data[0][1]
    assert len(t.data[0][1])==0, 'A zero threshold should give zero op types.'

  def test_full_threshold(self):
    cmd='native_op_profile --framework=tf --noplot --threshold 100 --print '+TESTFILE
    t = NativeOpProfileTool()
    runtool(t, cmd)
    print t.data[0][1]
    #assert len(t.data[0][1])==SIMPLE_NNET_OP_COUNT, 'Incorrectly thresholded some data vlues.'

    # FIXME: Currently the only assertion that was in this test as hard-coded,
    # which is extremely brittle. Find a better way.
    #opdiff = abs( len(t.data[0][1])-SIMPLE_NNET_OP_COUNT )
    #assert opdiff<OP_COUNT_THRESHOLD, 'Native op count in simple_nnet doesnt match: '+str(len(t.data[0][1]))+' vs '+str(SIMPLE_NNET_OP_COUNT)+'(expected)'
    assert len(t.data[0][1])>0, 'No ops found'


  def test_cachefile(self):
    with cleanup_cachefile(CACHEFILE):
      cmd = 'native_op_profile --cachefile='+str(CACHEFILE)+' --writecache '+TESTFILE
      t = NativeOpProfileTool()
      runtool(t, cmd)
      assert t.data[0][0]==TESTFILE, 'Corrupted cache: "'+str(t.data[0][0])+'"'

      cmd2 = 'native_op_profile --cachefile='+str(CACHEFILE)+' --readcache '
      t = NativeOpProfileTool()
      runtool(t, cmd2)
      assert t.data[0][0]==TESTFILE, 'Corrupted cache: "'+str(t.data[0][0])+'"'

