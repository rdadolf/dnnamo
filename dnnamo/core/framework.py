from abc import ABCMeta, abstractmethod
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

  @property
  def translator(self):
    '''Returns the instantiated, framework-specific translator object.'''

  ### Datatag accessors
  # Each of these methods corresponds to a Datatag, and their data is handled
  # by the DataManager.

  def _get_datatag(self, tag):
    tag.typecheck()
    if self._data_manager[tag] is None:
      self.collect(tag)
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
  # it bypasses the caching interface, which exists for a reason. If the user
  # really wants to force re-collecting data, then they should just invalidate
  # the data manager's cache and call the accessor again.

  def collect(self, datatag):
    if datatag.name=='graph':
      if datatag.ops=='native':
        if datatag.scope=='static':
          if datatag.mode=='training':
            self._data_manager[datatag] = self.model.get_training_graph()
          elif datatag.mode=='inference':
            self._data_manager[datatag] = self.model.get_inference_graph()
        elif datatag.scope=='dynamic':
          raise NotImplementedError, 'Dynamic graphs' # FIXME
      elif datatag.ops=='primitive':
        self._data_manager[datatag] = self.translator.translate(self.get_graph(datatag.mode))

    elif datatag.name=='weights':
      raise NotImplementedError, 'weights' # FIXME
      #self.model.get_weights() # FIXME: can't use this, need a mode switch

    elif datatag.name=='timing':
      raise NotImplementedError, 'timing' # FIXME

    elif datatag.name=='ivalues':
      raise NotImplementedError, 'ivalues' # FIXME

