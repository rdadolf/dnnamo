import numpy as np

import nnmodel
import nnmodel.frameworks
from nnmodel.core.mpl_plot import *
from tool_utilities import PlotTool

class Tool(PlotTool):
  TOOL_NAME='native_op_profile'
  TOOL_SUMMARY='Computes a breakdown of the time spent in each native operation type.'

  def add_subparser(self, argparser):
    super(Tool,self).add_subparser(argparser)
    self.subparser.add_argument('--print', action='store_true', default=False, help='Print a sorted list of the most time-consuming native operations.')
    self.subparser.add_argument('--threshold', action='store', type=float, default=101, help='Only consider the most time-consuming operations whose cumulative runtime is above a certain percentage of the total.')
    # comment this out
    # self.subparser.add_argument('--batch', action='store', type=int, default=32, help='Pass and sweep the batch size.')
    return self.subparser

  def _run(self, modelfiles):
    self.data = []
    for modelfile in modelfiles:
      Frame = nnmodel.frameworks.FRAMEWORKS[self.args['framework']]
      frame = Frame()
      # pass batch_size in here.. also add a command line for it.
      frame.load(modelfile, device='/cpu:0')
      #frame.load(modelfile, device='/cpu:0', init_options={'batch_size':self.args['batch']})
      #frame.load(modelfile)
      traces = frame.run_native_trace(n_steps=12, setup_options={'allow_soft_placement': True, 'inter_op_parallelism_threads':1, 'intra_op_parallelism_threads':8})
      #traces = frame.run_native_trace(n_steps=12, setup_options={'allow_soft_placement': True})
      # Hack off the first and last to avoid outliers.
      traces = traces[1:-1]
      self.data.append( [modelfile, self._aggregate_types(traces)] )

  def _aggregate_types(self, traces):
    breakdown = {} # op -> microseconds
    op_platform = {} # op -> microseconds
    for trace in traces:
      for op in trace:
        typ,dt,device = op.type,op.dt,op.device
        if typ not in breakdown:
          breakdown[typ] = 0
        breakdown[typ] += dt
        op_platform[typ] = device
    print op_platform.items()
    return breakdown

  def _threshold(self):
    new_self_data = []
    for modelfile,data in self.data:
      total_time = sum(data.values())
      print 'total_time',total_time
      threshold = total_time*self.args['threshold']/100.
      print 'threshold',threshold
      sorted_data = sorted(data.items(), key=lambda p:p[1], reverse=True)
      new_data = {}
      s = 0
      for typ,dt in sorted_data:
        if s>=threshold:
          break
        new_data[typ]=dt
        s += dt
      new_self_data.append([modelfile,new_data])
      #new_traces.append([modelfile,{typ:dt for typ,dt in data.items() if dt>lowerbound}])
    self.data = new_self_data

  def _output(self):
    #if self.args['threshold'] is not None:
    n_op_types_before = [len(data[1]) for data in self.data]
    self._threshold()
    n_op_types_after = [len(data[1]) for data in self.data]
    if self.args['print']:
      for ((modelfile,data),before,after) in zip(self.data,n_op_types_before,n_op_types_after):
        print '# of op types for '+str(modelfile)+': '+str(before)+'->'+str(after)
        for optype,dt in sorted(data.items(), key=lambda p:p[1], reverse=True):
          print '  '+str(optype)+': '+str(dt)

    if not self.args['noplot']:
      self._plot(self.args['plotfile'])

  def _plot(self, filename):
    fig,ax = plt.subplots(1,1,figsize=(12,9))
    modelfiles = [b[0] for b in self.data]
    colors = make_clr(len(modelfiles))
    for modelnum,(modelname,breakdown) in enumerate(self.data):
      names,counts = map(np.array,zip(*breakdown.items()))
      names,counts = zip(*sorted(zip(names,counts),key=lambda x:x[1],reverse=True))
      cum_counts=np.cumsum(counts)
      modname = modelname.split('/')[-1].split(':')[0]
      ax.plot(cum_counts, color=colors[modelnum], marker='.', label=modname, markersize=10)
      #ax.plot(cum_counts, color=colors[modelnum], marker='.', label=modelname, markersize=10)
      #ax.plot(100*cum_counts/np.sum(counts), color=colors[modelnum], marker='.', label=modelname, markersize=10)
    #ax.set_ylim(0,100)
    ax.set_ylabel('Cumulative time spent')
    ax.set_xlabel('Native operation types (sorted)')
    if True:
      lgd = ax.legend(loc='lower right', prop={'size':16})
      ax.set_yscale('log')
      #lgd = ax.legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.)
      fig.tight_layout()
      fig.savefig(filename, bbox_extra_artists=(lgd,), bbox_inches='tight')
    else:
      fig.tight_layout()
      fig.savefig(filename)
    return

    # FIXME: definitely needs work
    fig,ax = plt.subplots(1,1)
    modelfiles = [b[0] for b in self.data]
    maxcolors = max([len(b[1]) for b in self.data])
    colorsweep = make_clr(maxcolors)
    width=0.9
    for modelnum,(_,breakdown) in enumerate(self.data):
      if len(breakdown)!=0:
        names,counts = map(np.array,zip(*breakdown.items()))
        names,counts = zip(*sorted(zip(names,counts),key=lambda x:x[1],reverse=True))
        cum_counts=np.cumsum(counts)
        bottoms = np.hstack(([0],cum_counts[:-1]))
        for i,(n,c,b) in enumerate(zip(names,counts,bottoms)):
          #print(n,c,b)
          ax.bar(modelnum, c, bottom=b, width=width, label=n, color=colorsweep[i], lw=0.1)

    ax.set_xticks(np.arange(0,len(modelfiles))+width/2.)
    ax.set_xticklabels(modelfiles, rotation=89)
    ax.set_ylabel('Number of native ops')
    #fig.tight_layout()
    fig.savefig(filename)


