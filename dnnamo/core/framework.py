from abc import ABCMeta, abstractmethod, abstractproperty
from functools import wraps
import os
import timeit

from .model import DnnamoModel
from .datamanager import Datatag, DataManager

class _collector(object):
  # Decorator class for collector methods
  # Python does not syntactically support decorators referencing the parent
  # class of the methods they decorate (because the class is incompletely
  # defined at the time the decorator is called).
  # Python's inheritance system is also too limited to support the kind of
  # inheritance that we want to provide with Datatags.
  # This class is a way to get around both of those.
  # NOTE: if you create a subclass of Framework, you also need to subclass
  # this decorator, and it needs to copy the registry class attribute:
  #     class _my_collector(_collector): registry=dict(_collector.registry)
  # if you don't do this, you will not inherit the default collector methods.

  registry = {} # datatag => method

  def __init__(self, datatag):
    self._tag = datatag
  def __call__(self, method):
    self.__class__.registry[self._tag] = method
    return method


class Framework(object):
  __metaclass__ = ABCMeta

  # NOTE: Subclasses also need to have a similar statement to this one,
  #   except with their own _collector subclass.
  _collector_registry = _collector

  def __init__(self, loader=None, identifier=None, **kwargs):
    '''A general interface for interacting with neural networks.

    Args:
      loader: a DnnamoLoader class
      identifier: a string which the loader uses to load a model.
      kwargs: any additional arguments for the loader
    '''
    self._model = None
    self._data_manager = DataManager()
    if loader is not None:
      self.load(loader, identifier, **kwargs)

  def set_model(self, model):
    self._model = model
    # New model, so no chached data is valid anymore.
    self._data_manager.invalidate(Datatag('all','all','all','all'))

  def load(self, loader, identifier, **kwargs):
    '''Loads a model.

    The loader parameter expects a Dnnamo Loader class type.
    The identifier specifies which model to load, but its type depends on the
    loader that was selected.'''
    self._model = loader(identifier, **kwargs).load()
    if self._model is None:
      raise TypeError('No model returned by loader "'+str(loader.__class__.__name__)+'"')
    if not isinstance(self._model, DnnamoModel):
      raise TypeError('Invalid model returned by loader "'+str(loader.__class__.__name__)+'"')
    return self.model

  @property
  def model(self):
    '''Returns the current Dnnamo model.'''
    return self._model

  ### Framework-specific (abstract) functionality methods

  @abstractproperty
  def translator(self):
    '''Returns the instantiated, framework-specific translator object.'''

  @abstractproperty
  def ExemplarRegistry(self):
    '''Returns the Registry class for the exemplars for this framework.'''

  @abstractproperty
  def SyntheticModel(self):
    '''Returns a model class for building synthetic models from exemplar ops.'''

  ### Datatag accessors
  # Each of these methods corresponds to a Datatag, and their data is handled
  # by the DataManager.

  def _get_datatag(self, tag):
    tag.typecheck()
    if self._data_manager[tag] is None:
      self._collect(tag)
    return self._data_manager[tag]

  def get_graph(self, mode='training', scope='static', ops='native'):
    return self._get_datatag(Datatag('graph',mode=mode,scope=scope,ops=ops))

  def get_weights(self, mode='training'):
    return self._get_datatag(Datatag('weights',mode=mode,scope='static',ops='native'))

  def get_timing(self, mode='training', ops='native'):
    return self._get_datatag(Datatag('timing',mode=mode,scope='dynamic',ops=ops))

  def get_ivalues(self, mode='training'):
    return self._get_datatag(Datatag('ivalues',mode=mode,scope='dynamic',ops='native'))

  ### AMO methods

  def analyze(self, analysis):
    raise NotImplementedError

  ### Data collection
  # These are the methods actually used to collect data from the models.
  # It might seem a little redundant (why not just put this into the accessor),
  # but this gives us more flexibility. For instance, if a framework supports
  # a way of getting more than one piece of information at the same time, we can
  # fuse several of these collection methods into a single call which populates
  # the data manager's cache with the correct information. That way, if a user
  # calls get_timing() and then get_rungraph(), for example, the framework would
  # call the fused data collection routine on the first call, and the second
  # would return immediately using cached data.
  # 
  # It's probably obvious, but users should usually not be calling these, since
  # it bypasses the caching interface, which exists for a reason. If the user
  # really wants to force re-collecting data, then they should just invalidate
  # the data manager's cache and call the accessor again.

  @classmethod
  def _find_collector(cls, datatag):
    for k,v in cls._collector_registry.registry.items():
      if datatag in k.expand_mask():
        # NOTE: Returns the *first* collector---we do not support multiples
        return v

  def _collect(self, datatag):
    datatag.typecheck() # Only allow collecting exact datatags, not masks
    # In general, collecting two things at once is not usually correct.
    # If the user is using two things, they will use both accessors, which
    # will trigger both collectors.
    # If the framework wants to fuse two collectors, it should be using
    # more than one _collector decorator on the fused method, like this:
    #   @Framework._collector(Datatag1)
    #   @Framework._collector(Datatag2)
    #   def _collect_both_1_and_2_simultaneously(self, ...): ...
    f = self._find_collector(datatag)
    if f is not None:
      f(self,datatag)
    else:
      raise NotImplementedError('No framework method found to collect '+str(datatag))


  # Generic collector methods
  @_collector(Datatag('graph','all','all','primitive'))
  def _collect_primitive_graph(self, datatag):
    self._data_manager[datatag] = self.translator.translate(self.get_graph(mode=datatag.mode, scope=datatag.scope))

  @_collector(Datatag('weights','all','static','native'))
  def _collect_weights(self, datatag):
    # FIXME: we can't use this yet. We need a mode switch.
    # self.mode.get_weights()
    raise NotImplementedError('No framework method found to collect '+str(datatag))
