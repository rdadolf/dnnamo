import argparse
import os.path

# FIXME: add temporary directory context manager

def runtool(t, cmd):
  parser = argparse.ArgumentParser()
  subs = parser.add_subparsers()
  t.add_subparser(subs)
  args = vars(parser.parse_args(cmd.split()))
  print args
  t.run(args)

class cleanup_cachefile(object):
  def __init__(self,filename):
    self._cachefile = filename
  def __enter__(self):
    return self._cachefile
  def __exit__(self, *args):
    if os.path.exists(self._cachefile):
      assert os.path.isfile(self._cachefile), 'Corrupted cachefile path: '+str(self._cachefile)+' is not a regular file.'
      os.remove(self._cachefile)
