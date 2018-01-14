#!/usr/bin/env python

import numpy as np

import dnnamo
import dnnamo.frameworks
from dnnamo.core.mpl_plot import *
from .tool_utilities import PlotTool, ToolRegistry

class StationarityTool(PlotTool):
  TOOL_NAME='stationarity'
  TOOL_SUMMARY='Computes histogram of the total time across iterations'

  def add_subparser(self, argparser):
    super(StationarityTool,self).add_subparser(argparser)
    self.subparser.add_argument('--print', action='store_true', default=False, help='Print a sorted list of the most time-consuming native operations.')
    self.subparser.add_argument('--threshold', action='store', type=float, default=101, help='Only consider the most time-consuming operations whose cumulative runtime is above a certain percentage of the total.')
    return self.subparser

  def _run(self, models):
    self.data = []
    for model in models:
      Frame = dnnamo.frameworks.FRAMEWORKS[self.args['framework']]
      frame = Frame()
      frame.load(model, device='/cpu:0')
      #frame.load(model)
      traces = frame.run_native_trace(n_steps=102, setup_options={'allow_soft_placement': True})
      # Hack off the first and last to avoid outliers.
      traces = traces[1:-1]
      self.data.append( [model, [sum([tp.dt for tp in trace]) for trace in traces]] )
      #self.data.append( [model, self._aggregate_types(traces)] )

  #def _aggregate_types(self, traces):
    #breakdown = {} # op -> microseconds
    #op_platform = {} # op -> microseconds
    #for trace in traces:
    #  for op in trace:
    #    typ,dt,device = op.type,op.dt,op.device
    #    if typ not in breakdown:
    #      breakdown[typ] = 0
    #    breakdown[typ] += dt
    #    op_platform[typ] = device
    #print op_platform.items()
    #return breakdown

  #def _threshold(self):
  #  new_self_data = []
  #  for model,data in self.data:
  #    total_time = sum(data.values())
  #    print 'total_time',total_time
  #    threshold = total_time*self.args['threshold']/100.
  #    print 'threshold',threshold
  #    sorted_data = sorted(data.items(), key=lambda p:p[1], reverse=True)
  #    new_data = {}
  #    s = 0
  #    for typ,dt in sorted_data:
  #      if s>=threshold:
  #        break
  #      new_data[typ]=dt
  #      s += dt
  #    new_self_data.append([model,new_data])
  #    #new_traces.append([model,{typ:dt for typ,dt in data.items() if dt>lowerbound}])
  #  self.data = new_self_data

  def _output(self):
    #if self.args['threshold'] is not None:
    #n_op_types_before = [len(data[1]) for data in self.data]
    #self._threshold()
    #n_op_types_after = [len(data[1]) for data in self.data]
    if self.args['print']:
      for ((model,data),before,after) in zip(self.data,n_op_types_before,n_op_types_after):
        print '# of op types for '+str(model)+': '+str(before)+'->'+str(after)
        for optype,dt in sorted(data.items(), key=lambda p:p[1], reverse=True):
          print '  '+str(optype)+': '+str(dt)

    if not self.args['noplot']:
      self._plot(self.args['plotfile'])

  def _plot(self, filename):
    fig,ax = plt.subplots(1,1,figsize=(12,9))
    models = [b[0] for b in self.data]
    colors = make_clr(len(models))
    for modelnum,(modelname,data) in enumerate(self.data):
      ax.hist(np.array(data)+modelnum, label=modelname, color=colors[modelnum])
    #ax.set_ylim(0,100)
    #ax.set_ylabel('Cumulative time spent')
    xlims = ax.get_xlim()
    ax.set_xlim(0,xlims[1])
    ax.set_xlabel('Native operation types (sorted)')
    if False:
      lgd = ax.legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.)
      fig.tight_layout()
      fig.savefig(filename, bbox_extra_artists=(lgd,), bbox_inches='tight')
    else:
      fig.tight_layout()
      fig.savefig(filename)
    return

    # FIXME: definitely needs work
    fig,ax = plt.subplots(1,1)
    models = [b[0] for b in self.data]
    maxcolors = max([len(b[1]) for b in self.data])
    colorsweep = make_clr(maxcolors)
    width=0.9
    for modelnum,(_,data) in enumerate(self.data):
      if len(breakdown)!=0:
        names,counts = map(np.array,zip(*breakdown.items()))
        names,counts = zip(*sorted(zip(names,counts),key=lambda x:x[1],reverse=True))
        cum_counts=np.cumsum(counts)
        bottoms = np.hstack(([0],cum_counts[:-1]))
        for i,(n,c,b) in enumerate(zip(names,counts,bottoms)):
          #print(n,c,b)
          ax.bar(modelnum, c, bottom=b, width=width, label=n, color=colorsweep[i], lw=0.1)

    ax.set_xticks(np.arange(0,len(models))+width/2.)
    ax.set_xticklabels(models, rotation=89)
    ax.set_ylabel('Number of native ops')
    #fig.tight_layout()
    fig.savefig(filename)

ToolRegistry.register(StationarityTool)
