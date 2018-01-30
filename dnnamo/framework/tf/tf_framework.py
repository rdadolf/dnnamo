import tensorflow as tf

from ...core.datamanager import Datatag
from ...core.framework import Framework, _collector
from ...core.profile import Profile
from .tf_translator import TFTranslator
from .tf_graph import TFGraph

class _tf_collector(_collector):
  # This copies all of the collector methods from the core Framework registry.
  # This is a form of inheritance, which allows overriding collector methods.
  registry = dict(_collector.registry)

class TFFramework(Framework):
  _collector_registry = _tf_collector

  def __init__(self, model=None):
    super(TFFramework, self).__init__(model)
    self._translator = TFTranslator()

  @property
  def translator(self):
    return self._translator

  ### Collectors

  @_tf_collector(Datatag('graph','all','static','native'))
  def _collect_static_graphs(self, datatag):
    if datatag.mode=='training':
      g = TFGraph.from_graph(self.model.get_training_graph())
    elif datatag.mode=='inference':
      g = TFGraph.from_graph(self.model.get_inference_graph())
    else:
      raise TypeError, 'Invalid mode in datatag: '+str(datatag)
    self._data_manager[Datatag('graph',datatag.mode,'static','native')] = g

  @_tf_collector(Datatag('graph','all','dynamic','native'))
  @_tf_collector(Datatag('timing','all','dynamic','native'))
  def _collect_both_rungraph_and_timing(self, datatag):
    # Tensorflow's profiling collects everything at once, so we fill all relevant
    # datamanager cache fields at the same time whenever either is called. Saves
    # us from having to run thing twice.

    # Grab RunMetadata protobuf
    if datatag.mode=='training':
      rmds = self.model.profile_training(n_steps=10) # FIXME: how to set n_steps?
    elif datatag.mode=='inference':
      rmds = self.model.profile_inference(n_steps=10) # FIXME: how to set n_steps?
    else:
      raise KeyError, 'Invalid mode: '+str(mode)
    # Parse it into understandable data
    rungraph,timing = self._parse_rmd_proto(rmds)
    # Set the data manager's cache
    self._data_manager[Datatag('graph',datatag.mode,'dynamic','native')]=rungraph
    self._data_manager[Datatag('timing',datatag.mode,'dynamic','native')]=timing

  ### Internal

  def _parse_rmd_proto(self, rmds): # returns (<rungraph-object>, Profile)
    '''Extracts Dnnamo-relevant information from the RunMetadata protobuf.'''

    p = Profile()
    g = None # FIXME
    for rmd in rmds:
      # FIXME: investigate cost graph field of rmd?

      # Timing statistics
      #   Relevant TF protobuf definitions:
      #     tensorflow/core/protobuf/config.proto
      #     tensorflow/core/framework/step_stats.proto
      step_stats = rmd.step_stats
      for dev_step_stats in step_stats.dev_stats:
        device = dev_step_stats.device # string
        for node_stats in dev_step_stats.node_stats:
          name = node_stats.node_name
          dt = node_stats.all_end_rel_micros
          p.add(name,dt)

      # Run graph
      # FIXME
      _ = rmd.partition_graphs
      g = 'INVALID RUNGRAPH: THIS STRING IS USED TO TRICK THE DATAMANAGER'

    return (g,p)