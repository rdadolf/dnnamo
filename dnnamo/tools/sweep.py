
from .tool_utilities import AbstractTool, ToolRegistry
import numpy as np

from dnnamo.core.argsampler import UniformArgSampler
from dnnamo.framework.tf import TFFramework
from dnnamo.framework.tf.tf_exemplar import TFExemplarRegistry
from dnnamo.framework.tf.tf_synthesis import TFSyntheticModel

class SweepTool(AbstractTool):
  TOOL_NAME='sweep'
  TOOL_SUMMARY='Generating timing measurements suitable for training a performance estimator.'

  def add_subparser(self, argparser):
    super(SweepTool, self).add_subparser(argparser)
    self.subparser.add_argument('primop', type=str, help='Which primitive operation to sweep.')
    self.subparser.add_argument('-n', type=int, default=1, help='Set the PRNG seed for this sweep.')
    self.subparser.add_argument('--seed', type=int, default=None, help='Set the PRNG seed for this sweep.')
    # FIXME: Add argument to limit scope of sweep
    return self.subparser

  def _run(self):
    uas = UniformArgSampler()
    frame = TFFramework()
    primop_t = self.args['primop']
    primop_args = uas.sample(primop_t, n=self.args['n'], seed=self.args['seed'])
    Exemplar = TFExemplarRegistry.lookup(primop_t)
    self.data = []
    for p_args in primop_args:
      exemplar = Exemplar(p_args)
      synthmodel = TFSyntheticModel(exemplar)
      frame.set_model(synthmodel)
      profile = frame.get_timing(mode='inference', ops='native')
      graph = frame.get_graph(mode='inference', scope='dynamic', ops='native')
      id = graph.get_vertex_id_from_tf_name( exemplar.get_op_name() )
      self.data.append( (p_args, profile[id][0]) )

  def _output(self):
    print 'Output'
    for (args,time) in self.data:
      print args,':',time

ToolRegistry.register(SweepTool.TOOL_NAME, SweepTool)
