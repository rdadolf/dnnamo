from ...core.graph import DnnamoGraph
from .tf_op import TFOp
from .tf_tensor import TFTensor

class TFGraph(DnnamoGraph):

  # Constructor methods

  @classmethod
  def from_graph(cls, root_graph):
    g = TFGraph()

    for root_op in root_graph.get_operations():
      tf_op = TFOp.from_root_op(root_graph, root_op)
      g.add_op(tf_op)

    # Need to add all operations before any tensors.
    #for root_op in root_graph.get_operations():

      # All TensorFlow tensors are produced by a single operation, so
      # we don't need to worry about adding this tensor twice. (They can
      # be consumed twice, but that's not relevant here.)
      for root_tensor in root_op.values():
        tf_t = TFTensor.from_root_tensor(root_graph, root_tensor)
        g.add_tensor(tf_t)

    return g

  @classmethod
  def from_graphdef(cls, tf_graphdef):
    g = tf.Graph()
    with g.as_default():
      tf.import_graph_def(tf_graphdef)
    return cls.from_graph( g )


  def _check_dataflow_consistency(self, root_graph):
    assert len(self._ops)==len(root_graph.get_operations()), 'Mismatch in number of operations in graphs. TFGraph: '+str(len(self._ops))+' tf.Graph: '+str(len(root_graph.get_operations()))
    # Check all operations exist in both graphs
