from abc import ABCMeta, abstractmethod
import argparse

import dnnamo
from dnnamo.core.mpl_plot import *

class LoaderArgAction(argparse.Action):
  def __call__(self, parser, namespace, values, option_string):
    assert option_string=='--loader' or option_string=='-l', 'invalid use of the LoaderArgAction: '+str(option_string)
    setattr(namespace,'loader',getattr(dnnamo.loader,values))

class LoaderOptsArgAction(argparse.Action):
  def __init__(self, *args, **kwargs):
    super(LoaderOptsArgAction,self).__init__(*args, **kwargs)
    self.options = {}
  def __call__(self, parser, namespace, values, option_string):
    assert option_string=='--loader_opts', 'invalid use of the LoaderOptsArgAction: '+str(option_string)
    k,v = values.split('=')
    self.options[k] = v
    #print 'Total:',self.options
    setattr(namespace,'loader_opts', self.options)

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
    self.subparser.add_argument('models', type=str, nargs='*', help='a list of model identifiers. If there are multiple models, the tool will be run on each in the order specified.')
    self.subparser.add_argument('--framework', choices=dnnamo.frameworks.FRAMEWORKS.keys(), default='tf', help='specify which framework the models use')
    self.subparser.add_argument('--loader','-l', choices=dnnamo.loader.__all__, type=str, action=LoaderArgAction, default=dnnamo.loader.RunpyLoader, help='A Dnnamo loader class which will be used to load the model.')
    self.subparser.add_argument('--loader_opts', type=str, action=LoaderOptsArgAction, default={}, help='Specify additional options to the selected loader. Use the form key=value (no spaces).')
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
      self._run(self.args['models'])

    self._output()

    if self.args['writecache']:
      self._save(self.data, self.args['cachefile'])

  @abstractmethod
  def _output(self): pass

  @abstractmethod
  def _run(self, models): pass


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
