from ...core.graph import DnnamoGraph
from .tf_op import TFOp
from .tf_tensor import TFTensor

class TFGraph(DnnamoGraph):

  # Constructor/Augmentor methods

  @classmethod
  def from_graph(cls, root_graph):
    '''Creates a new TFGraph from a TensorFlow graph object.'''
    g = TFGraph()
    g.augment_from_graph(root_graph)
    return g

  def augment_from_graph(self, root_graph):
    '''Adds elements from a TensorFlow graph.'''
    for root_op in root_graph.get_operations():
      tf_op = TFOp.from_root_op(root_op)
      self.add_op(tf_op)
      # All TensorFlow tensors are produced by a single operation, so
      # we don't need to worry about adding this tensor twice. (They can
      # be consumed twice, but that's not relevant here.)
      for root_tensor in root_op.values():
        tf_t = TFTensor.from_root_tensor(root_tensor)
        self.add_tensor(tf_t)
    return self

  @classmethod
  def from_graphdef(cls, tf_graphdef):
    '''Creates a new TFGraph from a TensorFlow GraphDef object.'''
    g = tf.Graph()
    with g.as_default():
      tf.import_graph_def(tf_graphdef)
    return cls.from_graph( g )


  @classmethod
  def from_rmd(self, rmd):
    '''Creates a new TFGraph from a TensorFlow RunMetadata protobuf.'''
    g = TFGraph()
    g.augment_from_rmd(rmd)
    return g

  def augment_from_rmd(self, rmd):
    '''Adds elements from a TensorFlow RunMetadata protobuf.

    RunMetadata protobufs are the result of profiling the execution of a
    TensorFlow graph. This method makes a best-effort reconstruction of
    the graph that was run. The data TensorFlow provides can be incomplete.
    '''
    # At a high level, graph nodes (ops) are held in the 'partition_graph'
    # field, and memory/timing information is held in the 'step_stats' field.
    # Unfortunately, these structures are inconsistent at best. Fields often
    # do not exist, and many values must be inferred.

    ### First, collect operations from the partition graphs.

    # Segregate the pseudo-ops (any optype that begins with '_') into a
    # separate list for later.
    _transmission_ops = {} # _Send, _Recv
    _source_ops = {} # _Arg_* (XXX: Are these only placeholders or just anything that is replaced by a feed_dict argument?)
    _sink_ops = {} # _Result

    for part in rmd.partition_graphs:
      for node in part.node:
        if node.op.startswith('_'):
          pass # FIXME: Segregate
          print 'SEGREGATING',node
        else:
          tfop = TFOp.from_root_def(node)
          self.add_op(tfop)
    #for step
    # FIXME: NYI


  # Utility methdos

  def _check_dataflow_consistency(self, root_graph):
    # FIXME: This doesn't work for RMD graphs (since the root_graph is wrong)
    assert len(self._ops)==len(root_graph.get_operations()), 'Mismatch in number of operations in graphs. TFGraph: '+str(len(self._ops))+' tf.Graph: '+str(len(root_graph.get_operations()))
    # Check all operations exist in both graphs
