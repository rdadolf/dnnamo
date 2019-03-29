# pylint: disable=unused-import
# MPL Boilerplate
import json
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib.cm as cm
import matplotlib.colors
import matplotlib.patches
def rgb(r,g,b):
  return (float(r)/256.,float(g)/256.,float(b)/256.)
# Plot colors:
#   visually distinct under colorblindness and grayscale
crimson = rgb(172,63,64)
blue    = rgb(62,145,189)
teal    = rgb(98,189,153)
orange  = rgb(250,174,83)
#   luminance channel sweeps from dark to light, (for ordered comparisons)
clr = [crimson, blue, teal, orange]
def make_clr(n_colors):
  '''Return an array of n_colors colors, interpolated from the primary four.'''
  # Not the most efficient thing in the world...
  source_xs = np.arange(0,4)*n_colors/4.
  source_ys = zip(*[crimson,blue,teal,orange])
  dest_xs = np.arange(0,n_colors)
  dest_ys = np.array([np.interp(dest_xs,source_xs,source_y) for source_y in source_ys]).T.tolist()
  return dest_ys

mrk = ['o','D','^','s']
rcParams['figure.figsize'] = (8,6) # (w,h)
rcParams['figure.dpi'] = 150
# !$%ing matplotlib broke the interface. Why would you *replace* this!? >:(
try:
  from cycler import cycler
  rcParams['axes.prop_cycle'] = cycler('color',clr)
except ImportError:
  rcParams['axes.color_cycle'] = clr
rcParams['lines.linewidth'] = 2
rcParams['lines.marker'] = None
rcParams['lines.markeredgewidth'] = 0
rcParams['axes.facecolor'] = 'white'
rcParams['font.size'] = 22
rcParams['patch.edgecolor'] = 'black'
rcParams['patch.facecolor'] = clr[0]
rcParams['xtick.major.pad'] = 8
rcParams['xtick.minor.pad'] = 8
rcParams['ytick.major.pad'] = 8
rcParams['ytick.minor.pad'] = 8
#rcParams['font.family'] = 'Helvetica'
#rcParams['font.family'] = 'Liberation Sans'
rcParams['font.weight'] = 100

