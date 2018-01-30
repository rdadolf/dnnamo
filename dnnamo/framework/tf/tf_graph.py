from ...core.graph import DnnamoGraph
from .tf_op import TFOp
from ...core.tensor import DnnamoTensor

class TFGraph(DnnamoGraph):

  # Constructor methods

  @classmethod
  def from_graph(cls, tf_graph):
    g = TFGraph()
    # FIXME
    return g

  @classmethod
  def from_graphdef(cls, tf_graphdef):
    g = tf.Graph()
    with g.as_default():
      tf.import_graph_def(tf_graphdef)
    return cls.from_graph( g )
