import os.path
import runpy
import sys

from ..core.loader import BaseLoader

class RunpyLoader(BaseLoader):
  PROTECTED_FUNCTION_NAME='__dnnamo_loader__'

  def __init__(self, identifier, pypath=None):
    '''Loads a model implemented as a python file or module.

    The identifier parameter should be the name of a python module, as if it
    were being used with the "import" statement. If your model is a single
    file, the identifier is the filename without the ".py" extension. If the
    model is a module, it should be the module name. Note that this will use
    sys.path and PYTHONPATH as appropriate.

    The pypath is a shortcut for temporarily adding paths to sys.path. This
    argument is a single string or list of strings which will be added to
    sys.path for the purpose of loading this model only. The prior sys.path
    will not be modified.'''

    self.identifier = identifier
    if type(pypath)==str:
      self.pypath = [pypath]
    elif pypath is not None:
      self.pypath = [os.path.abspath(p) for p in pypath]
    else:
      self.pypath = []

  def load(self):
    old_syspath = sys.path
    sys.path[0:0] = self.pypath
    try:
      env = runpy.run_module(self.identifier)
    except ImportError as e:
      if len(self.pypath)>0:
        raise ImportError, 'Could not find a module named '+str(self.identifier)+' using the extra path '+str(self.pypath)
      else:
        raise ImportError, 'Could not find a module named '+str(self.identifier)
    sys.path = old_syspath
    try:
      model_function = env[self.PROTECTED_FUNCTION_NAME]
    except KeyError:
      raise NameError, 'no '+str(self.PROTECTED_FUNCTION_NAME)+' function found in module '+str(self.identifier)

    m = env[self.PROTECTED_FUNCTION_NAME]()
    print type(m)
    return m

