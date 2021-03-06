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
    try:
      k,v = values.split('=')
    except ValueError:
      raise ValueError('Must supply arguments to --loader_opts in the form "key=value". Got "'+str(values)+'" instead.')
    self.options[k] = v
    #print 'Total:',self.options
    setattr(namespace,'loader_opts', self.options)

################################################################################

class DataBlob(object):
  '''Stores data that a tool generates and provides an externalizable format.'''

  def __init__(self, keylist):
    self._d = {}
    for key in keylist:
      assert isinstance(key,str), 'Keylist must be a list of strings'
      self._d[key] = None

  def __getitem__(self, key):
    return self._d[key]

  def __setitem__(self, key, value):
    if key not in self._d:
      raise KeyError('"'+str(key)+'" is not a valid key. Must be one of: '+','.join([str(k) for k in self._d]))
    self._d[key] = value

  def to_json(self):
    return self._d

  def from_json(self, d):
    oldkeys = self._d.keys()
    newkeys = d.keys()
    for old in oldkeys:
      assert old in newkeys, 'Required key "'+old+'" not found in cache file.'
    for new in newkeys:
      assert new in oldkeys, 'Cache file key "'+new+'" not allowed in data.'
    self._d = d

################################################################################

class AbstractTool(object):
  __metaclass__ = ABCMeta

  TOOL_NAME = 'Tool'
  TOOL_SUMMARY = "A one-line description of the tool's function."
  CACHE_FORMAT = None # A list of key names for the data attribute.

  def __init__(self):
    self.subparser = None
    self.args = None
    assert hasattr(self, 'CACHE_FORMAT') and self.CACHE_FORMAT is not None, 'CACHE_FORMAT must be defined for this tool.'
    self.data = DataBlob(self.CACHE_FORMAT)

  def add_subparser(self, argparser):
    self.subparser = argparser.add_parser(self.TOOL_NAME, help=self.TOOL_SUMMARY)
    self.subparser.add_argument('--cachefile', metavar='PATH', type=str, default=self.TOOL_NAME+'.cache', help='Path where cache files are read from or written to.')
    self.subparser.add_argument('--readcache', action='store_true', default=False, help='Use data from a cache file instead of running.')
    self.subparser.add_argument('--writecache', action='store_true', default=False, help='Store data to a cache file after running.')
    return self.subparser

  def _save(self, filename=None):
    if filename is None:
      filename = self.TOOL_NAME+'.cache'
    with open(filename,'w') as fp:
      json.dump(self.data.to_json(), fp)
      print('File "'+filename+'" saved.')

  def _load(self, filename=None):
    if filename is None:
      filename = self.TOOL_NAME+'.cache'
    with open(filename,'r') as fp:
      self.data.from_json(json.load(fp))
      print('File "'+filename+'" loaded.')

  def run(self, args):
    # NOTE: if you override this, this line is generally still required.
    # You should also generally consider adding the other functionality, too.
    self.args = args

    if self.args['readcache']:
      self._load(self.args['cachefile'])
    else:
      v = self._run()
      if v is not None:
        return v

    self._output()

    if self.args['writecache']:
      self._save(self.args['cachefile'])

  @abstractmethod
  def _run(self): pass

  @abstractmethod
  def _output(self): pass

class BaselineTool(AbstractTool):
  __metaclass__ = ABCMeta

  def add_subparser(self, argparser):
    super(BaselineTool,self).add_subparser(argparser)
    self.subparser.add_argument('model', type=str, help='A model identifier. Different loaders understand different types of identifiers (paths, model names, etc.).')
    self.subparser.add_argument('--framework', choices=dnnamo.framework.FRAMEWORKS.keys(), default='tf', help='specify which framework the model uses')
    self.subparser.add_argument('--loader','-l', choices=dnnamo.loader.__all__, type=str, action=LoaderArgAction, default=dnnamo.loader.RunpyLoader, help='The Dnnamo loader class used to read in the model.')
    self.subparser.add_argument('--loader_opts', type=str, action=LoaderOptsArgAction, default={}, help='Additional options to the selected loader (key=value).')
    self.subparser.add_argument('--mode', choices=['training','inference'], type=str, default='training', help='Choose which mode the model will be analyzed in.')
    return self.subparser

  def run(self, args):
    # NOTE: if you override this, this line is generally still required.
    # You should also generally consider adding the other functionality, too.
    self.args = args

    if self.args['readcache']:
      self._load(self.args['cachefile'])
    else:
      v = self._run()
      if v is not None:
        return v

    self._output()

    if self.args['writecache']:
      self._save(self.args['cachefile'])

  @abstractmethod
  def _run(self): pass

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
