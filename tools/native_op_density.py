import numpy as np

import dnnamo
import dnnamo.frameworks
from dnnamo.core.mpl_plot import *
from .tool_utilities import PlotTool

class Tool(PlotTool):
  TOOL_NAME='native_op_density'
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
      frame.load(modelfile)
      m = frame.native_model()

      if self.args['framework']=='tf':
        self.data.append( [modelfile, self._tf_stats(m,frame)] )
      else:
        assert False, 'Unknown framework "'+str(self.args['framework'])+'"'

  def _tf_stats(self, m, frame):
    stats = frame.native_stats()
    retval = {}
    for op in m.get_operations():
      retval[op.name]['flops'], stats[op.name]['bytes'] = stats.computational_density(op.name)
    return retval

  def _plot(self, filename):
    fig,ax = plt.subplots(1,1,figsize=(12,9))
    modelfiles = [b[0] for b in self.data]
    colors = make_clr(len(modelfiles))
    # plot reference grid
    for abs_log2_density in xrange(0,4):
      a,b = 1, 1./(2**abs_log2_density)
      kwargs = {'color':(0.5,0.5,0.5), 'lw':0.5, 'transform':ax.transAxes}
      ax.plot([0,a],[0,b], **kwargs)
      ax.plot([0,b],[0,a], **kwargs)
    for modelnum,(modelname,op_stats) in enumerate(self.data):
      stats = op_stats.values()
      xs = np.array([s['bytes'] for s in stats])
      ys = np.array([s['flops'] for s in stats])
      ax.scatter(xs,ys,facecolor=colors[modelnum],lw=0,s=30,alpha=0.7,label=modelname)
    ax.set_ylabel('Flops')
    ax.set_xlabel('Bytes')
    maxlim = max(ax.get_xlim()[0], ax.get_ylim()[1])
    ax.set_xlim(0,maxlim)
    ax.set_ylim(0,maxlim)
    if True:
      lgd = ax.legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.)
      fig.tight_layout()
      fig.savefig(filename, bbox_extra_artists=(lgd,), bbox_inches='tight')
    else:
      fig.tight_layout()
      fig.savefig(filename)

