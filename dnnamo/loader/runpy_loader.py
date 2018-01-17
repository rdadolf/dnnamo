import os.path
import runpy
import sys

from ..core.loader import BaseLoader

class RunpyLoader(BaseLoader):
  PROTECTED_FUNCTION_NAME='__dnnamo_loader__'

  def __init__(self, identifier):
    '''Loads a model implemented as a python file or module.

    The identifier parameter should be the name or path to a python module.
    The RunpyLoader will attempt to disambiguate the identifier based on the
    following rules:
      - If the identifier ends with ".py", it will be treated as a path to
        a python script file.
      - If the identifier points to an existing path location, it will be
        treated as a path to a python module. The prefix will be temporarily
        added to PYTHONPATH and the suffix will be loaded as a module (as if
        using the "import" statement).
      - If none of these apply, the identifier will be loaded as a module (as if
        using the "import" statement) with no modification to PYTHONPATH.
    For the latter two cases, sys.path and PYTHONPATH will be used as appropriate.'''

    self.identifier = identifier

  def _id_to_loadpair(self, ident):
    '''Converts an identifier into a (module, pypath) pair.'''
    if ident.endswith('.py'):
      prefix,suffix = os.path.split(os.path.abspath(ident))
      return (suffix[:-3], [prefix])
    elif os.path.exists(ident):
      prefix,suffix = os.path.split(os.path.abspath(ident))
      return (suffix, [prefix])
    else:
      return (ident, [])

  def load(self):
    (module,pypath) = self._id_to_loadpair(self.identifier)
    old_syspath = sys.path
    sys.path[0:0] = pypath
    try:
      env = runpy.run_module(module)
    except ImportError:
      # FIXME: This reports a generic import error even if the module *IS*
      #   found, if the module itself raises an exception. This exception causes
      #   the module import to fail, which raises an ImportError from runpy.
      #   We need a better way to disambiguate the two cases.
      if len(pypath)>0:
        raise ImportError, 'Error while finding or loading module named '+str(module)+' using the extra path '+str(pypath)
      else:
        raise ImportError, 'Error while finding or loading module named '+str(module)
    sys.path = old_syspath

    # This try block doesn't create the model, because we only want to capture
    # KeyErrors from the env lookup. If the user's load function throws one, we
    # want those to pass through.
    try:
      _ = env[self.PROTECTED_FUNCTION_NAME]
    except KeyError:
      raise NameError, 'no '+str(self.PROTECTED_FUNCTION_NAME)+' function found in module '+str(module)

    m = env[self.PROTECTED_FUNCTION_NAME]()
    return m

