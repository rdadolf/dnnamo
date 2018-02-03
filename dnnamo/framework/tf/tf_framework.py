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

  def __init__(self, loader=None, identifier=None, **kwargs):
    super(TFFramework, self).__init__(loader, identifier, **kwargs)
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

  #def _fix_nodes(self, nodes):
  #  return [_ for _ in map(self._fix_node,nodes) if _ is not None]

  #def _fix_node(self, nodedef):
  #  # TF uses the _ to signal invalid ops (since ops are required to conform
  #  #   to a specific Regex defined in tensorflow/python/framework/ops.py.
  #  if nodedef.name.startswith('_'):
  #    # Return value nodes can be eliminated
  #    if nodedef.op=='_Retval':
  #      return None
  #    # Argument placeholder arguments can be replaced with tf.Placeholder ops
  #    elif nodedef.name.startswith('_arg_Placeholder') and (nodedef.op=='_Arg'):
  #      return tf.NodeDef(
  #        name = nodedef.name[5:],
  #        op = u'Placeholder',
  #        input = nodedef.input,
  #        device = nodedef.device,
  #        attr = nodedef.attr
  #      )
  #    else:
  #      raise NotImplementedError('Unknown operation: '+str(nodedef.name)+' : '+str(nodedef.op))
  #      #return None # Eliminate other operations
  #  else:
  #    # Known ops are unchanged.
  #    return nodedef

  def _parse_rmd_proto(self, rmds): # returns (<rungraph-object>, Profile)
    '''Extracts Dnnamo-relevant information from the RunMetadata protobuf.'''

    # FIXME: Graph Aggregation
    #for rmd in rmds:
    rmd = rmds[0]

    p = Profile()
    # TODO: investigate cost graph field of rmd?

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
    raise NotImplementedError('Waiting on TFGraph.from_rmd() method')

    #unified_rungraph = tf.Graph()
    #with unified_rungraph.as_default():
      # Unify parts
      #for part in rmd.partition_graphs:
      #
      #  valid_part = tf.GraphDef(
      #    versions = part.versions,
      #    node = self._fix_nodes(part.node)
      #  )
      #  tf.import_graph_def(valid_part, name='')
    #g = TFGraph.from_graph(unified_rungraph) # FIXME

    assert False
    return (g,p)
