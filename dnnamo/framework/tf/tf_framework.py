from ...core.bimap import Bimap
from ...core.datamanager import Datatag
from ...core.framework import Framework, _collector
from ...core.profile import Profile
from .tf_exemplar import TFExemplarRegistry
from .tf_graph import TFGraph
from .tf_synthesis import TFSyntheticModel
from .tf_translator import TFTranslator

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

  @property
  def ExemplarRegistry(self):
    return TFExemplarRegistry

  @property
  def SyntheticModel(self):
    return TFSyntheticModel

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

  @_tf_collector(Datatag('timing','all','dynamic','primitive'))
  def _translate_timing_information(self, datatag):
    # Collect requisite data
    g_prim = self.get_graph(mode=datatag.mode, scope='dynamic', ops='primitive')
    p_nat = self.get_timing(mode=datatag.mode, ops='native')
    # Compute a name map between native and primitive op ID's
    op_names = Bimap()
    for primop in g_prim.ops:
      op_names.l[primop.id] = primop.root.id
    # Create the primitive timing profile
    p_prim = Profile()
    for natop_id,timing_list in p_nat.items():
      p_prim[op_names.r[natop_id]] = timing_list

    self._data_manager[datatag] = p_prim

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
    rmd = rmds[0]

    # TODO: investigate cost graph field of rmd?

    # Run graph
    g = TFGraph.from_rmd(rmd)

    # Timing statistics
    p = Profile()
    #   Relevant TF protobuf definitions:
    #     tensorflow/core/protobuf/config.proto
    #     tensorflow/core/framework/step_stats.proto
    step_stats = rmd.step_stats
    for dev_step_stats in step_stats.dev_stats:
      # device = dev_step_stats.device # string
      for node_stats in dev_step_stats.node_stats:
        name = node_stats.node_name
        # NOTE: We remove _SOURCE ops, since they are treated bizarrely by
        # TensorFlow: they never have dataflow dependencies, they never exist
        # in the runtime graph, and yet they have non-zero compute time.
        # Luckily, there is only one _SOURCE op in an RMD structure, so its
        # contribution to overall computation time is negligible (and does not
        # scale with any quantity)
        if name=='_SOURCE':
          continue
        id = g.get_vertex_id_from_tf_name(name)
        dt = node_stats.all_end_rel_micros
        p.add(id,dt)

    return (g,p)
