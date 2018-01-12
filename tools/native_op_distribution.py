import numpy as np

import dnnamo
import dnnamo.frameworks
from dnnamo.loader import RunpyLoader
from dnnamo.core.mpl_plot import *
from .tool_utilities import PlotTool, path_to_loader_pair

class Tool(PlotTool):
  TOOL_NAME='native_op_distribution'
  TOOL_SUMMARY='Computes a breakdown of the counts of each native operation type in a model.'

  def __init__(self):
    super(Tool,self).__init__()

  def add_subparser(self, argparser):
    super(Tool,self).add_subparser(argparser)
    return self.subparser

  def _run(self, modelfiles):
    self.data = []
    for modelfile in modelfiles:
      Frame = dnnamo.frameworks.FRAMEWORKS[self.args['framework']]
      frame = Frame()
      modname, pypath = path_to_loader_pair(modelfile)
      frame.load(RunpyLoader, modname, pypath=pypath)

      if self.args['framework']=='tf':
        self.data.append( [modelfile, self._tf_breakdown(frame.graph)] )
      else:
        assert False, 'Unknown framework "'+str(self.args['framework'])+'"'

  def _plot(self, filename):
    fig,ax = plt.subplots(1,1,figsize=(12,9))
    modelfiles = [b[0] for b in self.data]
    colors = make_clr(len(modelfiles))
    for modelnum,(modelname,breakdown) in enumerate(self.data):
      names,counts = map(np.array,zip(*breakdown.items()))
      names,counts = zip(*sorted(zip(names,counts),key=lambda x:x[1],reverse=True))
      cum_counts=np.cumsum(counts)
      ax.plot(cum_counts, color=colors[modelnum], marker='.', label=modelname, markersize=10)
    ax.set_ylabel('Total number of native operations')
    ax.set_xlabel('Native operation types (sorted)')
    if False:
      lgd = ax.legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.)
      fig.tight_layout()
      fig.savefig(filename, bbox_extra_artists=(lgd,), bbox_inches='tight')
    else:
      fig.tight_layout()
      fig.savefig(filename)

  def _tf_breakdown(self, graph):
    breakdown = {}
    for op in graph.get_operations():
      typ = op.type
      if typ not in breakdown:
        breakdown[typ] = 0
      breakdown[typ] += 1
    return breakdown
