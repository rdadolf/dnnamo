from abc import ABCMeta, abstractmethod
from .model import BaseModel
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
      if not isinstance(model, BaseModel):
        raise TypeError, 'Must supply a Dnnamo Model instance.'
    self._model = model
    self._data_manager = DataManager()

    self._traces = None
    self._stats = None

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

  @property
  def graph(self):
    '''Returns the underlying computational graph.'''
    return self.model.get_graph()

  ### Framework-specific (abstract) functionality methods

  @property
  def translator(self):
    '''Returns the instantiated, framework-specific translator object.'''

  ### Datatag accessors
  # Each of these methods corresponds to a Datatag, and their data is handled
  # by the DataManager.

  @property
  def absgraph(self):
    '''Returns an Abstract Graph representation of the model.'''
    if self._data_manager[Datatag.absgraph] is None:
      self._data_manager[Datatag.absgraph] = self.translator.translate(self.model)
    return self._data_manager[Datatag.absgraph]

  def get_weights(self, selector=None):
    '''Returns a dictionary of weights values from the model.

    The optional selector argument allows callers to specify a class which can
    filter only certain weights to return.'''
    if self._data_manager[Datatag.weights] is None:
      self._data_manager[Datatag.weights] = self.model.get_weights()
    return self._data_manager[Datatag.weights]

  ### AMO methods

  def analyze(self, analysis, trigger='demand'):
    raise NotImplementedError

  ### Others

  # FIXME: Not sure I need this
  # It *is* currently used to do one-off lookups for primop-native_op checks in
  # tools/direct_reconstruction. There's probably a way to avoid having these
  # two methods by making a better translation interface (or adding some things
  # to AbstractGraph's in order to sidestep the need for doing those lookups.
  def translate_native_op(self, native_op_id):
    '''Returns the primop id(s) corresponding to a given native op.'''
    assert self._translator is not None, 'No translator constructed. This is a bug.'
    return self._translator.map_native_op(native_op_id)
  # FIXME: Same as above.
  def translate_primop(self, primop_id):
    '''Returns the native op id(s) corresponding to a given primop.'''
    assert self._translator is not None, 'No translator constructed. This is a bug.'
    return self._translator.map_primop(primop_id)


  def run(self, n_steps=1, setup_options=None):
    '''Executes a model for a finite number of steps.

    Args:
      n_steps: The number of steps to execute.
    '''
    #self._model.setup(setup_options=setup_options)
    runner = self.DefaultRunstep()
    # FIXME: When the Model interface changed, run() was split into run_train()
    #   and run_inference(). This function used to wrap run(), but it cannot
    #   anymore. In order to maintain things in a semblence of working order, I
    #   substituted run_inference.
    #   In the end, this function should probably be split into two. Or, better
    #   yet, it should be redesigned into something less broad ("run" does just
    #   about everything, and maybe it shouldn't).
    self.model.run_inference(runstep=runner, n_steps=n_steps)
    #self._model.teardown()
    return None

  def run_native_trace(self, n_steps=1, setup_options=None):
    '''Executes a model, capturing a trace of its execution.

    Args:
      n_steps: The number of steps to execute.

    Returns:
      traces: An array of Trace objects, one for each step executed.
    '''
    #self._model.setup(setup_options=setup_options)
    runner = self.InstrumentedRunstep()
    # FIXME: When the Model interface changed, run() was split into run_train()
    #   and run_inference(). This function used to wrap run(), but it cannot
    #   anymore. In order to maintain things in a semblence of working order, I
    #   substituted run_inference.
    self.model.run_inference(runstep=runner, n_steps=n_steps)
    #self._model.teardown()
    self._traces = runner.traces
    return self._traces

  ### Internal utility methods


  @abstractmethod
  def _build_native_stats(self, graph, traces):
    '''Create a new Stats object, collecting data if necessary.'''

  def native_stats(self):
    '''Returns a (possibly cached) Stats object.'''
    if self._stats is None:
      assert self.model is not None, 'Must load a model before collecting statistics'
      if self._traces is None:
        self.run_native_trace(n_steps=12)[1:-1]
      self._stats = self._build_native_stats( self.graph, self._traces )
    return self._stats
