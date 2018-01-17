import tensorflow as tf

from dnnamo.core.framework import Framework
from dnnamo.frameworks.tf.tf_translator import TFTranslator

class TFFramework(Framework):
  def __init__(self, model=None):
    super(TFFramework, self).__init__(model)
    self._translator = TFTranslator()

  @property
  def translator(self):
    return self._translator

  def collect_rungraph(self):
    raise NotImplementedError

  def collect_timing(self):
    raise NotImplementedError

  ### Internal

  def _parse_rmd_proto(self, rmd):
    '''Extracts Dnnamo-relevant information from the RunMetadata protobuf.'''
    g = rmd.partition_graphs
    stats = rmd.step_stats
    # FIXME: investigate cost graph field of rmd

