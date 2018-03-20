
from .tool_utilities import AbstractTool, ToolRegistry

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
    self.subparser.add_argument('--seed', type=int, help='Set the PRNG seed for this sweep.')
    # FIXME: Add argument to limit scope of sweep
    return self.subparser

  def _run(self):
    uas = UniformArgSampler()
    frame = TFFramework()
    primop_t = self.args['primop']
    # FIXME: loop
    primop_args = uas.sample(primop_t)[0] # Fixme: sample N times
    print primop_args
    Exemplar = TFExemplarRegistry.lookup(primop_t)
    exemplar = Exemplar(primop_args)
    synthmodel = TFSyntheticModel(exemplar)
    frame.set_model(synthmodel)
    print 'Running'
    profile = frame.get_timing(mode='inference', ops='native')
    print 'Exemplar op name:',synthmodel.get_exemplar_op_name()
    print profile.items()

  def _output(self):
    print 'Output'

ToolRegistry.register(SweepTool.TOOL_NAME, SweepTool)
