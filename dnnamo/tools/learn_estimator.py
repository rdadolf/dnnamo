
from .tool_utilities import AbstractTool, ToolRegistry
import numpy as np

from dnnamo.core.estimator import EstimatorRegistry
from dnnamo.core.features import Features
from dnnamo.framework.tf import TFFramework

class LearnEstimatorTool(AbstractTool):
  TOOL_NAME='learn-estimator'
  TOOL_SUMMARY='Train a performance estimator on timing measurements.'

  def add_subparser(self, argparser):
    super(LearnEstimatorTool, self).add_subparser(argparser)
    self.subparser.add_argument('primop', type=str, help='Which primitive operation to learn.')
    self.subparser.add_argument('features', type=str, nargs='+', help='Data files to concatenate and use to train the estimator.')
    self.subparser.add_argument('-e', '--estimator', type=str, default='ols', choices=EstimatorRegistry.keys(), help='Estimator to learn.')
    self.subparser.add_argument('-o', '--output', type=str, default=None, help='Set the output filename (default: <primop>-<estimator>.est)')
    return self.subparser

  def _run(self):
    if self.args['output'] is None:
      self.args['output'] = str(self.args['primop'])+'-'+str(self.args['estimator'])+'.est'
    primop_t = self.args['primop']
    EstimatorClass = EstimatorRegistry.lookup(self.args['estimator'])
    est = EstimatorClass()

    f = Features()
    for feats_file in self.args['features']:
      f2 = Features()
      f2.read(feats_file)
      f.concatenate(f2)

    est.fit(f)
    self.data = est.get_params()
    self._save(self.data, self.args['output'])

  def _output(self):
    print 'Model arguments:'
    print ','.join(str(arg) for arg in self.data)

ToolRegistry.register(LearnEstimatorTool.TOOL_NAME, LearnEstimatorTool)
