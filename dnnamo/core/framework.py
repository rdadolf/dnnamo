from abc import ABCMeta, abstractmethod
from .model import BaseModel, ImmutableModel, StaticModel, DynamicModel
from .manager import AnalysisManager, TransformManager

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
      assert isinstance(model, BaseModel), 'Must supply a Model instance.'
    self._model = model
    self._analysis_manager = AnalysisManager()
    self._transform_manager = TransformManager()

    self._absgraph = None

    self._translator = None

    self._traces = None
    self._stats = None

  def load(self, loader, identifier, **kwargs):
    '''Loads a model.

    The loader parameter expects a Dnnamo Loader class type.
    The identifier specifies which model to load, but its type depends on the
    loader that was selected.'''
    self._model = loader(identifier, **kwargs).load()
    return self._model

  @property
  def graph(self):
    '''Returns the underlying computational graph.'''
    return self._model.get_graph()

  @property
  def model(self):
    '''Returns the current Dnnamo model.'''
    return self._model

  def translate_native_op(self, native_op_id):
    '''Returns the primop id(s) corresponding to a given native op.'''
    assert self._translator is not None, 'No translator constructed. This is a bug.'
    return self._translator.map_native_op(native_op_id)

  def translate_primop(self, primop_id):
    '''Returns the native op id(s) corresponding to a given primop.'''
    assert self._translator is not None, 'No translator constructed. This is a bug.'
    return self._translator.map_primop(primop_id)

  @abstractmethod
  def absgraph(self):
    '''Returns an Abstract Graph representation of the model.'''
    return self._absgraph

  def analyze(self, analysis, trigger='demand'):
    raise NotImplementedError

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
    self._model.run_inference(runstep=runner, n_steps=n_steps)
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
    self._model.run_inference(runstep=runner, n_steps=n_steps)
    #self._model.teardown()
    self._traces = runner.traces
    return self._traces

  @abstractmethod
  def _build_native_stats(self, native_model, traces):
    '''Create a new Stats object, collecting data if necessary.'''

  def native_stats(self):
    '''Returns a (possibly cached) Stats object.'''
    if self._stats is None:
      assert self._model is not None, 'Must load a model before collecting statistics'
      if self._traces is None:
        self.run_native_trace(n_steps=12)[1:-1]
      self._stats = self._build_native_stats( self.graph, self._traces )
    return self._stats

