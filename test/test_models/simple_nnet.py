import numpy as np
import tensorflow as tf

from dnnamo.core.model import DnnamoModel
from dnnamo.framework.tf.tf_model import session_run, session_profile

class SimpleNNet(DnnamoModel):
  def __init__(self):
    super(SimpleNNet,self).__init__()
    self.session = None
    self.g = tf.Graph()
    self.minibatch = 32
    self.trainsize = 2048
    #with self.g.device(device):
    self._create_graph()
    # NOTE This used to be called by an external entity (probably a Framework),
    #   but we removed setup() and teardown() from the Model interface. To
    #   simplify the port, we kept the old setup() function in this file, made it
    #   internal, and call it only once now.
    #   In this test instance, I don't think this causes problems, but in general
    #   it is not necessarily safe.
    self._setup()

  def _create_graph(self):
    # 100:10 NN
    with self.g.as_default():
      self.input = tf.placeholder(tf.float32, shape=[None, 100])
      self.labels = tf.placeholder(tf.float32, shape=[None, 10])
      glorot_bounds = np.sqrt(3.)/np.sqrt(10.+10.)
      self.W = tf.Variable(tf.random_uniform(shape=[100,10], minval=-glorot_bounds, maxval=glorot_bounds), name='W')
      self.b = tf.Variable(tf.zeros(shape=[10]), name='b')
      self.inference = tf.nn.softmax( tf.add(tf.matmul(self.input, self.W, name='matmul'), self.b, name='sum'), name='inference' )

      self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits( logits=self.inference, labels=self.labels ), name='loss')
      self.train = tf.train.GradientDescentOptimizer(1.0).minimize(self.loss)

  def get_training_graph(self):
    return self.g

  def get_inference_graph(self):
    return self.g

  def get_weights(self, keys=None):
    # FIXME: implement key filtering
    return {'W': self.W, 'b': self.b}

  def set_weights(self, kv):
    # FIXME: THIS IS COMPLETELY UNTESTED.
    #   I think the idea is right, but I'm almost certain it's buggy.
    for k,v in {'W':self.W, 'b':self.b}:
      try:
        kv[k]
        with self.g.as_default():
          assign_op = tf.assign(v, kv[k])
          with tf.Session() as s:
            s.run(assign_op)
      except KeyError: pass
    return True

  def _setup(self, setup_options=None):
    with self.g.as_default():
      if setup_options is not None:
        self.session = tf.Session(config=tf.ConfigProto(**setup_options))
      else:
        self.session = tf.Session()
      # FIXME: Deprecated initializer
      self.session.run(tf.global_variables_initializer())
      # Random training data
      self.example_data = np.random.rand(self.trainsize,100)
      self.example_labels = np.random.rand(self.trainsize,10)

  def _inference(self, profile, n_steps=1, *args, **kwargs):
    b0,b1 = 0, n_steps%self.trainsize
    outs = []
    fetches = [self.inference]
    feed_dict = {self.input: self.example_data[b0:b1]}
    if profile:
      out = session_profile(self.session, fetches=fetches, feed_dict=feed_dict)
    else:
      out = session_run(self.session, fetches=fetches, feed_dict=feed_dict)
    outs.append(out)
    return outs

  def run_inference(self, *args, **kwargs): # pylint: disable=W0221
    return self._inference(False, *args, **kwargs)

  def profile_inference(self, *args, **kwargs): # pylint: disable=W0221
    return self._inference(True, *args, **kwargs)

  def _training(self, profile, n_steps=1, *args, **kwargs):
    b0,b1 = 0,self.minibatch
    outs = []
    for _ in range(0,n_steps):
      b0 = (b0+self.minibatch)%self.trainsize
      b1 = min( (b0+self.minibatch), self.trainsize)
      fetches = [self.train, self.loss]
      feed_dict = {self.input: self.example_data[b0:b1], self.labels: self.example_labels[b0:b1]}
      if profile:
        out = session_profile(self.session, fetches=fetches, feed_dict=feed_dict)
      else:
        _,loss = session_run(self.session, fetches=fetches, feed_dict=feed_dict)
        out = (loss, None)
      outs.append(out)
    return outs

  def run_training(self, *args, **kwargs): # pylint: disable=W0221
    return self._training(False, *args, **kwargs)

  def profile_training(self, *args, **kwargs): # pylint: disable=W0221
    return self._training(True, *args, **kwargs)

  def get_intermediates(self, *args, **kwargs):
    # FIXME: left blank pending a definition of "activation" (c.f. core/model.py)
    return dict()

def __dnnamo_loader__():
  return SimpleNNet()
