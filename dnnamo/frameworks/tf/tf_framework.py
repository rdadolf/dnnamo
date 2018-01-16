import tensorflow as tf

from dnnamo.core.framework import Framework
from dnnamo.frameworks.tf.tf_translator import TFTranslator
from dnnamo.frameworks.tf.tf_runstep import _DefaultRunstep, _InstrumentedRunstep
from dnnamo.frameworks.tf.tf_stats import TFNativeStats

class TFFramework(Framework):
  def __init__(self, model=None):
    super(TFFramework, self).__init__(model)
    self._translator = TFTranslator()

  @property
  def translator(self):
    return self._translator

  # FIXME:
  # def run_instrumented(self, ...)
  # Tensorflow returns captures rungraph and timing data using the same method:
  # Session.run() with runmetadata. Instead of running this twice, just run it
  # once whenever either is asked for and fill in both fields. This also makes
  # the data more consistent.
  # Note that we might also need to override get_timing() and get_rungraph() in
  # order to make this work.

  ### Older methods

  # This is a convenient shortcut so users don't have to go module surfing.
  DefaultRunstep = _DefaultRunstep
  InstrumentedRunstep = _InstrumentedRunstep

  def _build_native_stats(self, graph, traces):
    return TFNativeStats(graph, traces)
