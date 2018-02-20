from abc import ABCMeta, abstractmethod
import argparse

import dnnamo
from .mpl_plot import *

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
    self.subparser.add_argument('models', type=str, nargs='*', help='A list of model identifiers. Different loaders understand different types of identifiers (paths, model names, etc.).')
    self.subparser.add_argument('--framework', choices=dnnamo.framework.FRAMEWORKS.keys(), default='tf', help='specify which framework the models use')
    self.subparser.add_argument('--loader','-l', choices=dnnamo.loader.__all__, type=str, action=LoaderArgAction, default=dnnamo.loader.RunpyLoader, help='The Dnnamo loader class used to read in the model.')
    self.subparser.add_argument('--loader_opts', type=str, action=LoaderOptsArgAction, default={}, help='Additional options to the selected loader (key=value).')
    self.subparser.add_argument('--cachefile', metavar='PATH', type=str, default=self.TOOL_NAME+'.cache', help='Path where cache files are read from or written to.')
    self.subparser.add_argument('--readcache', action='store_true', default=False, help='Use data from a cache file instead of running.')
    self.subparser.add_argument('--writecache', action='store_true', default=False, help='Store data to a cache file after running.')
    return self.subparser

  def run(self, args):
    # NOTE: if you override this, this line is generally still required.
    # You should also generally consider adding the other functionality, too.
    self.args = args

    if self.args['readcache']:
      self.data = self._load(self.args['cachefile'])
    else:
      if len(self.args['models'])<1:
        print 'No models selected.'
        return
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


class ToolRegistry(dnnamo.core.registry.Registry):
  @classmethod
  def sorted_tools(cls):
    dev_sorted = [(k,v) for k,v in sorted(cls.items(), key=lambda pair:pair[0]) if k.startswith('_')]
    normal_sorted = [(k,v) for k,v in sorted(cls.items(), key=lambda pair:pair[0]) if not k.startswith('_')]
    return normal_sorted + dev_sorted
