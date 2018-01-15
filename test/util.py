import argparse
import os
import os.path
import tempfile
import shutil

# FIXME: add temporary directory context manager

def runtool(t, cmd):
  parser = argparse.ArgumentParser()
  subs = parser.add_subparsers()
  t.add_subparser(subs)
  args = vars(parser.parse_args(cmd.split()))
  print args
  return t.run(args)


class cleanup_cachefile(object):
  def __init__(self,filename):
    self._cachefile = filename
  def __enter__(self):
    return self._cachefile
  def __exit__(self, *args):
    if os.path.exists(self._cachefile):
      assert os.path.isfile(self._cachefile), 'Corrupted cachefile path: '+str(self._cachefile)+' is not a regular file.'
      os.remove(self._cachefile)

class in_temporary_directory(object):
  def __init__(self):
    self._prefix = 'dnnamo_test_dir_'
  def __enter__(self):
    self._d = os.path.abspath(tempfile.mkdtemp(prefix=self._prefix))
    self._old_d = os.getcwd()
    os.chdir(self._d)
    print 'Working in temporary directory ',self._d
    return self._d
  def __exit__(self, *args):
    os.chdir(self._old_d)
    shutil.rmtree(self._d)
