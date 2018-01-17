from abc import ABCMeta, abstractmethod
from functools import wraps
import os
import timeit

from .model import DnnamoModel
from .datamanager import Datatag, DataManager

def _datatag_accessor(function):
  @wraps(function)
  def wrapper(*args, **kwargs):
    if 'mode' in kwargs:
      if kwargs['mode'] not in ['training','inference']:
        raise KeyError, 'mode argument must be either training or inference'
    return function(*args, **kwargs)
  return wrapper

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
    # Keep separate data for separate modes (since the graphs can be different)
    self._data_manager = {'training': DataManager(), 'inference': DataManager()}


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

  @property
  def translator(self):
    '''Returns the instantiated, framework-specific translator object.'''

  ### Datatag accessors
  # Each of these methods corresponds to a Datatag, and their data is handled
  # by the DataManager.

  @_datatag_accessor
  def get_graph(self, mode='training'):
    '''Returns the underlying computational graph.'''
    if self._data_manager[mode][Datatag.graph] is None:
      self.collect_graph(mode=mode)
    return self._data_manager[mode][Datatag.graph]

  @_datatag_accessor
  def get_absgraph(self, mode='training'):
    '''Returns an Abstract Graph representation of the model.'''
    if self._data_manager[mode][Datatag.absgraph] is None:
      self.collect_absgraph(mode=mode)
    return self._data_manager[mode][Datatag.absgraph]

  @_datatag_accessor
  def get_weights(self, selector=None, mode='training'):
    '''Returns a dictionary of weights values from the model.

    The optional selector argument allows callers to specify a class which can
    filter only certain weights to return.'''
    if self._data_manager[mode][Datatag.weights] is None:
      self.collect_weights(mode=mode)
    return self._data_manager[mode][Datatag.weights]

  @_datatag_accessor
  def get_rungraph(self, mode='training'):
    '''Returns the native graph actually used during execution.

    Note that depending on the framework and framework settings used, this may
    or may not be identical to the graph return via the Framework.graph property.'''
    if self._data_manager[mode][Datatag.rungraph] is None:
      self.collect_rungraph(mode=mode)
    return self._data_manager[mode][Datatag.rungraph] # FIXME: what does this return?

  @_datatag_accessor
  def get_timing(self, mode='training'):
    '''Return timing information for all native operations.'''
    if self._data_manager[mode][Datatag.timing] is None:
      self.collect_timing(mode=mode)
    return self._data_manager[mode][Datatag.timing] # Profile object

  ### AMO methods

  def analyze(self, analysis, trigger='demand'):
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
  # it bypasses the caching interface, which exists for a reason. We reinforce
  # this by not returning anything from these functions. If the user really
  # wants to force re-collecting data, then they should just invalidate the
  # data manager's cache and call the accessor again.

  def collect_graph(self, mode='training'):
    if mode=='training':
      self._data_manager['training'][Datatag.graph] = self.model.get_training_graph()
    elif mode=='inference':
      self._data_manager['inference'][Datatag.graph] = self.model.get_inference_graph()
    else:
      raise KeyError, 'Invalid mode: '+str(mode)

  def collect_absgraph(self, mode='training'):
    self._data_manager[mode][Datatag.absgraph] = self.translator.translate(self.get_graph(mode))

  def collect_weights(self, mode='training'):
    self._data_manager[Datatag.weights] = self.model.get_weights()

  @abstractmethod
  def collect_rungraph(self, mode='training'):
    raise NotImplementedError

  @abstractmethod
  def collect_timing(self, mode='training'):
    raise NotImplementedError
