from abc import ABCMeta, abstractmethod
import os.path

import dnnamo
from dnnamo.core.mpl_plot import *

def path_to_loader_pair(path):
  '''Converts a path to a (module, pypath) pair suitable for runpy.

  Assuming that a path points to a valid python file or module which contains a
  Dnnamo model that can be loaded with RunpyLoader.'''

  p = os.path.normpath(path) # mostly so "/path/dir/" becomes "/path/dir"
  pypath, modname = os.path.split(p)
  if modname.endswith('.py'):
    modname = modname[:-3]
  return (modname, pypath)

class BaselineTool(Cacher):
  __metaclass__ = ABCMeta

  TOOL_NAME = 'Tool'
  TOOL_SUMMARY = 'A tool for Dnnamo.'

  def __init__(self):
    self.subparser = None
    self.args = None
    self.data = None

  def add_subparser(self, argparser):
    self.subparser = argparser.add_parser(self.TOOL_NAME, help=self.TOOL_SUMMARY)
    self.subparser.add_argument('modelfiles', type=str, nargs='*', help='a list of model files or modules. If there are multiple models, add the name of the desired model after a ":".')
    self.subparser.add_argument('--framework', choices=dnnamo.frameworks.FRAMEWORKS.keys(), default='tf', help='specify which framework the models use')
    self.subparser.add_argument('--cachefile', metavar='PATH', type=str, default=self.TOOL_NAME+'.cache', help='location for reading or writing pre-computed data')
    self.subparser.add_argument('--readcache', action='store_true', default=False, help='Do not run any computation. Use data from a cache file instead.')
    self.subparser.add_argument('--writecache', action='store_true', default=False, help='Run the computation and store the result to a cache file.')
    return self.subparser

  def run(self, args):
    # Note: if you override this, this line is generally still required.
    self.args = args
    if self.args['readcache']:
      self.data = self._load(self.args['cachefile'])
    else:
      self._run(self.args['modelfiles'])

    self._output()

    if self.args['writecache']:
      self._save(self.data, self.args['cachefile'])

  @abstractmethod
  def _output(self): pass

  @abstractmethod
  def _run(self, modelfiles): pass


class PlotTool(BaselineTool):
  def add_subparser(self, argparser):
    super(PlotTool, self).add_subparser(argparser)
    self.subparser.add_argument('--plotfile', metavar='PATH', type=str, default=self.TOOL_NAME+'.pdf', help='Location to store the output plot.')
    self.subparser.add_argument('--noplot', action='store_true', default=False, help='Do not plot any data. Just run the tool quietly.')
    return self.subparser

  def _output(self):
    if not self.args['noplot']:
      self._plot(self.args['plotfile'])

  @abstractmethod
  def _plot(self, filename): pass


class ToolRegistry(object):
  registry = {}

  @classmethod
  def register(cls, ToolClass):
    cls.registry[ToolClass.TOOL_NAME] = ToolClass

  @classmethod
  def all_tools(cls):
    return cls.registry.items()
