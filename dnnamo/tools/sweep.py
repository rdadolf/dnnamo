from dnnamo.core.argsampler import UniformArgSampler
from dnnamo.core.features import Features
from dnnamo.framework.tf import TFFramework

from .tool_utilities import AbstractTool, ToolRegistry

class SweepTool(AbstractTool):
  TOOL_NAME='sweep'
  TOOL_SUMMARY='Generating timing measurements suitable for training a performance estimator.'
  CACHE_FORMAT=[] # FIXME: Features are not cached.

  def add_subparser(self, argparser):
    super(SweepTool, self).add_subparser(argparser)
    self.subparser.add_argument('primop', nargs='?', type=str, help='Which primitive operation to sweep.')
    self.subparser.add_argument('-n', type=int, default=1, help='Number of data points to collect.')
    self.subparser.add_argument('-o', '--output', type=str, default=None, help='Output file with Estimator training features (default: <primop>.features).')
    self.subparser.add_argument('--seed', type=int, default=None, help='Set the PRNG seed for this sweep.')
    self.subparser.add_argument('-l','--list', default=False, action='store_true', help="Don't actually run a sweep, just print which primops are supported,")
    # FIXME: Add argument to limit scope of sweep
    return self.subparser

  def _run(self):
    frame = TFFramework()
    if self.args['list']:
      print 'Supported Exemplars:'
      for primop in frame.ExemplarRegistry:
        print ' ',primop
      return 0
    elif self.args['primop'] is None:
      self.subparser.error('A primop must be specified unless using --list')

    primop_t = self.args['primop']
    uas = UniformArgSampler()
    primop_args = uas.sample(primop_t, n=self.args['n'], seed=self.args['seed'])
    Exemplar = frame.ExemplarRegistry.lookup(primop_t)
    self.features = Features()
    for p_args in primop_args:
      exemplar = Exemplar(p_args)
      synthmodel = frame.SyntheticModel(exemplar)
      frame.set_model(synthmodel)
      profile = frame.get_timing(mode='inference', ops='native')
      graph = frame.get_graph(mode='inference', scope='dynamic', ops='native')
      id = graph.get_vertex_id_from_tf_name( exemplar.get_op_name() )
      self.features.append( p_args, profile[id][0] )

    if self.args['output'] is None:
      self.args['output'] = str(primop_t)+'.features'
    self.features.write(self.args['output'])

  # FIXME: Features objects cannot be cached in the normal way.

  #def _load(self, filename):
  #  feats = Features()
  #  feats.read(filename)
  #  return feats

  #def _save(self, data, filename): # pylint: disable=W0221
  #  data.write(filename)

  def _output(self):
    print 'Output'
    for args,time in zip(self.features.op_arguments, self.features.measurements):
      print args,':',time

ToolRegistry.register(SweepTool.TOOL_NAME, SweepTool)
