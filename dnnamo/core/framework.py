from abc import ABCMeta, abstractmethod, abstractproperty
from functools import wraps
import os
import timeit

from .model import DnnamoModel
from .datamanager import Datatag, DataManager

class Framework(object):
  __metaclass__ = ABCMeta

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
    return self._get_datatag(Datatag('graph',mode=mode,scope='static',ops='native'))

  def get_timing(self, mode='training', ops='native'):
    return self._get_datatag(Datatag('graph',mode=mode,scope='dynamic',ops=ops))

  def get_ivalues(self, mode='training'):
    return self._get_datatag(Datatag('graph',mode=mode,scope='dynamic',ops='native'))

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

  _collector_methods = {}

  @classmethod
  def _collector(cls,datatag):
    '''Decorator for collector methods'''
    if len(cls._collector_methods)==0: # only true once, ever.
      cls._populate_default_collectors()
    def tag_override_method(function):
      cls._collector_methods[datatag] = function
      return function
    return tag_override_method

  @classmethod
  def _find_collector(cls, datatag):
    for k,v in cls._collector_methods.items():
      if datatag in k.expand_mask():
        # NOTE: Returns the *first* collector---we do not support multiples
        return v

  def _collect(self, datatag):
    datatag.typecheck() # Only allow collecting exact datatags, not masks
    # In general, collecting two things at once is not usually correct.
    #
    # If the user is using two things, they will use both accessors, which
    # will trigger both collectors.
    #
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

  # Python has no good way of calling decorators with access to the class
  # they're defined in. I.E.
  # this code...                   ...becomes this code:
  #   class Foo(object):              class Foo(object):
  #     def decorator(...): ...         def decorator(...): ...
  #     @decorator                      def _func(self, ...): ...
  #     def method(self, ...): ...      method = decorator(_func):
  # Which means there's no good way to write a decorator that takes a
  # class argument, because this is illegal:
  #   class Foo(object):
  #     def decorator(cls, ...): ...
  #     def _func(self, ...): ...
  #     method = Foo.decorator(_func) # Can't reference Foo from within Foo
  #
  # So instead, we try to sidestep the problem by populating the
  # _collector_methods dictionary manually the first time they're accessed.
  # XXX: I am not *entirely* sure that populating the dictionary with
  #   unbound methods (i.e., cls._collect_...) and them calling them as
  #   bound methods (i.e., f(self,datatag)) is actually correct. But it
  #   seems to be working. Unfortunately, we have to keep the populate
  #   function a @classmethod, since it is called by subclasses as a
  #   decorator: @Framework._collector(...).
  @classmethod
  def _populate_default_collectors(cls):
    if len(cls._collector_methods)==0:
      cls._collector_methods = {
        Datatag('graph','training','static','native'):
          cls._collect_training_graph_from_model,
        Datatag('graph','inference','static','native'):
          cls._collect_inference_graph_from_model,
        Datatag('graph','all','static','primitive'):
          cls._collect_primitive_graph,
        Datatag('weights','all','static','native'):
          cls._collect_weights,
      }

  # Generic collector methods
  def _collect_training_graph_from_model(self, datatag):
    self._data_manager[datatag] = self.model.get_training_graph()

  def _collect_inference_graph_from_model(self, datatag):
    self._data_manager[datatag] = self.model.get_inference_graph()

  def _collect_primitive_graph(self, datatag):
    self._data_manager[datatag] = self.translator.translate(self.get_graph(datatag.mode))

  def _collect_weights(self, datatag):
    # FIXME: we can't use this yet. We need a mode switch.
    # self.mode.get_weights()
    raise NotImplementedError('No framework method found to collect '+str(datatag))
