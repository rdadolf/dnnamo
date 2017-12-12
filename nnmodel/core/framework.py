from abc import ABCMeta, abstractmethod
from .model import Model

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
      assert isinstance(model, Model), 'Must supply a Model instance.'
    self._model = model
    self._graph = None
    self._dgraph = None
    self._translator = None
    self._traces = None
    self._stats = None

  def _split_modelname(self, filename):
    ''' '''
    (prefix,_,suffix) = str(filename).rsplit(':')
    return (prefix,suffix)

  @abstractmethod
  def load(self, filename, **kwargs):
    '''Loads a model from a file.'''

  def native_model(self):
    '''Extract a native model representation.'''
    return self._model.model()

  def model(self):
    '''Returns a reference to an NNModel Frame object wrapping the native model.'''
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
  def graph(self):
    '''Returns a DGraph.'''

  def run(self, n_steps=1, setup_options=None):
    '''Executes a model for a finite number of steps.

    Args:
      n_steps: The number of steps to execute.
    '''
    self._model.setup(setup_options=setup_options)
    runner = self.DefaultRunstep()
    self._model.run(runstep=runner, n_steps=n_steps)
    self._model.teardown()
    return None

  def run_native_trace(self, n_steps=1, setup_options=None):
    '''Executes a model, capturing a trace of its execution.

    Args:
      n_steps: The number of steps to execute.

    Returns:
      traces: An array of Trace objects, one for each step executed.
    '''
    self._model.setup(setup_options=setup_options)
    runner = self.InstrumentedRunstep()
    self._model.run(runstep=runner, n_steps=n_steps)
    self._model.teardown()
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
      self._stats = self._build_native_stats( self.model(), self._traces )
    return self._stats

