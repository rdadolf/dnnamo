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

  def __init__(self, model=None):
    '''Frameworks can be built from several input sources.

    Args:
      model: A Model instance of the appropriate type.
             (This is not common---it is usually easier to pull the model
              from a file than to build it dynamically.)
    '''
    if model is not None:
      if not isinstance(model, DnnamoModel):
        raise TypeError, 'Must supply a Dnnamo Model instance.'
    self._model = model
    self._data_manager = DataManager()


  def load(self, loader, identifier, **kwargs):
    '''Loads a model.

    The loader parameter expects a Dnnamo Loader class type.
    The identifier specifies which model to load, but its type depends on the
    loader that was selected.'''
    self._model = loader(identifier, **kwargs).load()
    return self.model

  @property
  def model(self):
    '''Returns the current Dnnamo model.'''
    return self._model

  ### Framework-specific (abstract) functionality methods

  @abstractproperty
  def translator(self):
    '''Returns the instantiated, framework-specific translator object.'''

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
  @_collector(Datatag('graph','training','static','native'))
  def _collect_training_graph_from_model(self, datatag):
    self._data_manager[datatag] = self.model.get_training_graph()

  @_collector(Datatag('graph','inference','static','native'))
  def _collect_inference_graph_from_model(self, datatag):
    self._data_manager[datatag] = self.model.get_inference_graph()

  @_collector(Datatag('graph','all','static','primitive'))
  def _collect_primitive_graph(self, datatag):
    self._data_manager[datatag] = self.translator.translate(self.get_graph(datatag.mode))

  @_collector(Datatag('weights','all','static','native'))
  def _collect_weights(self, datatag):
    # FIXME: we can't use this yet. We need a mode switch.
    # self.mode.get_weights()
    raise NotImplementedError('No framework method found to collect '+str(datatag))
