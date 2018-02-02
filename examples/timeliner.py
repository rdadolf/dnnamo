#!/usr/bin/env python

import dnnamo
import tensorflow as tf
import sys

from tensorflow.python.client import timeline

def main(modelfile):
  frame = dnnamo.framework.tf.TFFramework(dnnamo.loader.RunpyLoader, modelfile)

  rmds = frame.model.profile_training(n_steps=10)

  trace = timeline.Timeline(rmds[0].step_stats)
  with open('timeline.ctf.json','w') as trace_file:
    trace_file.write(trace.generate_chrome_trace_format())

  print 'Timeline file in "timeline.ctf.json"'  

if __name__=='__main__':
  for modelfile in sys.argv[1:]:
    main(modelfile)
