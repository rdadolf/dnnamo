import timeit

from ..core.profile import Profile
from ..framework import FRAMEWORKS
from .tool_utilities import BaselineTool, ToolRegistry

class MimicTool(BaselineTool):
  TOOL_NAME='mimic'
  TOOL_SUMMARY='Attempts to estimate the performance of a neural network by decomposing it into abstract pieces and then reassembling it.'
  CACHE_FORMAT=[
    'wall_time', # Total wall clock time of the model
    'true_time', # Total reconstructed time of the profile
    'true_profile', # Profiled op times
    'mimic_time', # Total reconstructed time of the mimic
    'mimic_profile', # Mimicked op times
  ]

  def __init__(self):
    super(MimicTool,self).__init__()

  def add_subparser(self, argparser):
    super(MimicTool,self).add_subparser(argparser)
    self.subparser.add_argument('--detail','-d', default='profile', choices=['profile','primop','regression'], help='The level of abstraction to which the neural net is decomposed before being reassembled.')
    self.subparser.add_argument('--full', default=False, action='store_true', help='Prints timing information for each individual component being estimated.')
    return self.subparser

  def _run(self):
    model = self.args['model']
    frame = FRAMEWORKS[self.args['framework']]()
    frame.load(self.args['loader'], model, **self.args['loader_opts'])

    actions = { 'profile': self._mimic_profile,
                'primop': self._mimic_primop,
                'regression': self._mimic_regression }

    t0 = timeit.default_timer()
    if self.args['mode']=='inference':
      _ = frame.model.run_inference(n_steps=1)
    elif self.args['mode']=='training':
      _ = frame.model.run_training(n_steps=1)
    else:
      raise ValueError('Invalid mode: '+str(self.args['mode']))
    t1 = timeit.default_timer()
    self.data['wall_time'] = (t1-t0)*1000000. # to microseconds
    actions[self.args['detail']](frame)

  def _output(self):
    super(MimicTool,self)._output()

    wall_time = self.data['wall_time']
    true_time = self.data['true_time']
    #true_profile = self.data['true_profile']
    mimic_time = self.data['mimic_time']
    mimic_profile = self.data['mimic_profile']

    def fmt(t): return str(round(t,2)) if t is not None else 'None'

    print('Wall Time: '+fmt(wall_time))
    print('True Time: '+fmt(true_time))
    print('Mimic Time: '+fmt(mimic_time))

    if self.args['full']:
      print 'Component timing information:'
      # FIXME: include true profile information
      if mimic_profile is None:
        print '  No components found.' # mostly for empty cachefiles
        return
      for name,timing in mimic_profile.items():
        print '  '+str(name)+':',timing

  def _aggregate_profile(self, profile):
    return sum([usecs for _,usecs in profile.aggregate('last').items()])

  def _mimic_profile(self, frame):
    '''Mimic a model by aggregating pointwise native op measurements.'''

    timing_info = frame.get_timing(self.args['mode'], ops='native')
    t_sum = self._aggregate_profile(timing_info)
    self.data['true_time'] = t_sum
    self.data['true_profile'] = timing_info

  def _mimic_primop(self, frame):
    '''Mimic a model by synthesizing and running primops matching the model.

    For each op, we translate it into a primop, then use the primop parameters
    to synthesize an exemplar op and run it.'''
    # Grab all of the primops
    pgraph = frame.get_graph(self.args['mode'], ops='primitive', scope='dynamic')
    actual_timing = frame.get_timing(self.args['mode'], ops='primitive')
    primops = pgraph.ops

    # For each, create and run a synthetic model
    exemplar_registry = frame.ExemplarRegistry
    timing_info = Profile()
    for primop in primops:
      if primop.type in exemplar_registry:
        exemplar = exemplar_registry.lookup(primop.type)(primop.argvalues)
        synthmodel = frame.SyntheticModel(exemplar)
        frame.set_model(synthmodel)
        profile = frame.get_timing(mode='inference', ops='native')
        graph = frame.get_graph(mode='inference', scope='dynamic', ops='native')
        id = graph.get_vertex_id_from_tf_name( exemplar.get_op_name() )
        # FIXME: This ID is a bit ambiguous: do we want the id of the original
        # native op, the id of the primop, or the id of the exemplar?
        timing_info.extend( primop.id, profile[id] )
        # FIXME: Remove this debug info
        s = str(primop.type)+'<'+','.join([str(_) for _ in primop.argvalues])+'>'
        s+= ': '+str(profile[id][0])+' vs. '+str(actual_timing[primop.id][0])+' '
        s+= str(primop.root.id)+'('+str(primop.root.type)
        if hasattr(primop,'broadcast'):
          s+=str('-cast')
        s+=')'
        print(s)
      else:
        #print('SKIPPED: '+str(primop)+' '+str(actual_timing[primop.id]))
        pass

    #t_sum = self._aggregate_profile(timing_info)

    # FIXME: Need to assign values to cachefile
    #return (t_sum, timing_info)

  def _mimic_regression(self, frame):
    '''Mimic a model using a pre-trained regression model.

    For each op, we translate it into a primop, then feed its parameters into
    a pre-trained model to estimate the op's behavior.'''

    # FIXME: Need to assign values to cachefile
    #return (0,None)

ToolRegistry.register(MimicTool.TOOL_NAME, MimicTool)
