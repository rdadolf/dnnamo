#!/usr/bin/env python
import argparse
import os.path
import importlib

import numpy as np

import nnmodel
import nnmodel.frameworks
from nnmodel.core.mpl_plot import *
from tool_utilities import PlotTool

class Tool(PlotTool):
  TOOL_NAME='heavy_tail'
  TOOL_SUMMARY='Estimates the heavy-tailed-ness of the time distribution for a model.'

  def add_subparser(self, argparser):
    super(Tool,self).add_subparser(argparser)
    self.subparser.add_argument('--print', action='store_true', default=False, help='Print a sorted list of the most time-consuming native operations.')
    return self.subparser

  def _run(self, modelfiles):
    self.data = []
    for modelfile in modelfiles:
      Frame = nnmodel.frameworks.FRAMEWORKS[self.args['framework']]
      frame = Frame()
      frame.load(modelfile)
      traces = frame.run_native_trace(n_steps=12)
      # Hack off the first and last to avoid outliers.
      traces = traces[1:-1]
      self.data.append( [modelfile, self._aggregate_types(traces)] )

  def _aggregate_types(self, traces):
    breakdown = {} # op -> microseconds
    for trace in traces:
      for op in trace:
        typ,dt = op.type,op.dt
        if typ not in breakdown:
          breakdown[typ] = 0
        breakdown[typ] += dt
    return breakdown

  def _output(self):
    if not self.args['noplot']:
      self._plot(self.args['plotfile'])

  def _plot(self, filename):
    fig,ax = plt.subplots(1,1,figsize=(12,9))
    #modelfiles = [b[0] for b in self.data]
    #colors = make_clr(len(modelfiles))
    epsilon = .000000001
    for _,(_,breakdown) in enumerate(self.data):
      # Plot survivor probability where survival%(x) === % ops with dt>x
      op_dt = np.sort(np.array(breakdown.values()))

      cum_dt = np.cumsum(op_dt)
      n = op_dt.shape[0]
      cum_prob = np.arange(n-1,-1.,-1)
      s = float(np.sum(cum_prob))
      print s
      s = np.trapz(cum_prob, cum_dt)
      print s
      cum_prob /= s

      # Tack on a dt=0 point and strip the last point (where survival==0)
      c_dt = np.insert(cum_dt,0, 0)[:-1]
      c_prob = np.insert(cum_prob, 0, n)[:-1]

      ax.plot( c_dt, c_prob, marker='.', markersize=10 )

      from scipy.stats import expon, pareto, lognorm
      for distr in [expon,pareto,lognorm]:
        params = distr.fit(op_dt)
        print params
        xs = np.arange(cum_dt[0],cum_dt[-1])
        ys = distr.pdf(xs,*params)
        ax.plot( xs, ys )

      #params = expon.fit(op_dt[30:])
      #print params
      #xs = np.arange(cum_dt[0],cum_dt[-1])
      #ys = expon.pdf(xs,*params)
      #ax.plot( xs, ys )

      #params = pareto.fit(op_dt[30:])
      #print params
      #xs = np.arange(cum_dt[0],cum_dt[-1])
      #ys = pareto.pdf(xs,*params)
      #ax.plot( xs, ys )
      
    #ax.set_xscale('log')
    #ax.set_yscale('log')
    ax.set_ylim(epsilon,.00001)
    #ax.set_ylim(epsilon,.01)
    ax.set_xlim(0,c_dt[-1])
    ax.set_ylabel('Probability of op type lasting >=dt')
    ax.set_xlabel('dt')
    if False:
      lgd = ax.legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.)
      fig.tight_layout()
      fig.savefig(filename, bbox_extra_artists=(lgd,), bbox_inches='tight')
    else:
      fig.tight_layout()
      fig.savefig(filename)
    return
