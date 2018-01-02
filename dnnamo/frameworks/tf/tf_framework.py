import sys
import os.path
import imp
import inspect
import random

import tensorflow as tf

from dnnamo.core.framework import Framework
from dnnamo.core.model import BaseModel, ImmutableModel, StaticModel, DynamicModel
from dnnamo.frameworks.tf.tf_translator import TFTranslator
from dnnamo.frameworks.tf.tf_runstep import _DefaultRunstep, _InstrumentedRunstep
from dnnamo.frameworks.tf.tf_stats import TFNativeStats

class TFFramework(Framework):
  def __init__(self, model=None):
    super(TFFramework, self).__init__(model)
    self._translator = TFTranslator()

  def graph(self):
    if self._dgraph is None:
      assert self._model is not None, 'No model loaded. Run load(...) first.'
      self._dgraph = self._translator.translate( self._model )
    return self._dgraph

  def _transitive_closure(self, targets):
    # NOTE: Operational, but not currently used.
    #   Part of the problem is the need for a 'targets' argument, which is a
    #   model-specific runtime parameter. This not usually convenient to get
    #   when we would most want a static transitive closure function. I'm
    #   leaving it in here in case it is useful later.
    ops = set([])
    op_queue = []
    # Prime the queue
    for t in targets:
      if isinstance(t, tf.Tensor):
        #print 'adding new op',t.op.name
        op_queue.append(t.op)
        ops.add(t.op)
      else:
        #print 'adding new op',t.name
        op_queue.append(t)
        ops.add(t)
    # BFS the graph
    while len(op_queue)>0:
      op = op_queue[0]
      op_queue = op_queue[1:]
      for pre_op in [tensor.op for tensor in op.inputs]:
        if pre_op not in ops:
          #print 'adding new op',pre_op.name
          ops.add(pre_op)
          op_queue.append(pre_op)
      for pre_op in op.control_inputs:
        if pre_op not in ops:
          #print 'adding new control op',pre_op.name
          ops.add(pre_op)
          op_queue.append(pre_op)
    # All the ops necessary for computing the targets
    return list(ops)

  # This is a convenient shortcut so users don't have to go module surfing.
  DefaultRunstep = _DefaultRunstep
  InstrumentedRunstep = _InstrumentedRunstep

  def _build_native_stats(self, native_model, trace):
    return TFNativeStats(native_model, trace)
