#!/usr/bin/env python

import dnnamo
import pdb
import tensorflow as tf
import sys

from tensorflow.python.client import timeline

def main(modelfile):
  frame = dnnamo.framework.tf.TFFramework(dnnamo.loader.RunpyLoader, modelfile)

  rmds = frame.model.profile_training(n_steps=10)
  rmd = rmds[0]

  pdb.set_trace()

  print 'Timeline file in "timeline.ctf.json"'  

if __name__=='__main__':
  for modelfile in sys.argv[1:]:
    main(modelfile)
