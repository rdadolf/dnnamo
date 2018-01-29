import tensorflow as tf

from ...core.op import DnnamoOp

class TFOp(DnnamoOp):
  def __init__(self, tf_op_or_nodedef):
    self._root = tf_op_or_nodedef
    self._id = None
    self._optype = None
    self._params = None
    try:
      self._from_op(tf_op_or_nodedef)
    except Exception as nodedef_error:
      try:
        self._from_nodedef(tf_op_or_nodedef)
      except Exception as op_error:
        print 'Raised exception treating as a NodeDef: '+str(nodedef_error)
        print 'Raised exception treating as a tf.Operation: '+str(op_error)
        raise TypeError('Unrecognized native operation '+str(tf_op_or_nodedef))

  def _from_nodedef(self, nodedef):
    # TF op names are already unique
    # NOTE: If we start creating TFOp ops independent of an actual tf.Graph
    #   context, then it's possible that the tf.Operation names will not be
    #   unique (since they're not in the same graph namespace). If that
    #   happens, then we should implement an internal id counter like the one
    #   in the primop class. The names won't line up directly anymore, but we
    #   can always retrieve them from the source op.
    self._id = nodedef.name
    self._optype = nodedef.op
    self._params = None # FIXME

  def _from_op(self, op):
    # TF op names are already unique
    # NOTE: If we start creating TFOp ops independent of an actual tf.Graph
    #   context, then it's possible that the tf.Operation names will not be
    #   unique (since they're not in the same graph namespace). If that
    #   happens, then we should implement an internal id counter like the one
    #   in the primop class. The names won't line up directly anymore, but we
    #   can always retrieve them from the source op.
    self._id = op.name
    self._optype = op.type
    self._params = None # FIXME




  @property
  def id(self):
    return self._id

  @property
  def optype(self):
    return self._optype

  @property
  def parameter_names(self):
    return self._params.keys()

  @property
  def parameter_values(self):
    return self._params.values()

  @property
  def parameters(self):
    return self._params.items()

  @property
  def root(self):
    return self._root

  def __str__(self):
    return '<TFOp_'+str(self.optype)+':'+str(self.id)+'>'
