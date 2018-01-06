from dnnamo.core.model import DynamicModel

import tensorflow as tf
import numpy as np

class SimpleNNet(DynamicModel):
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
      self.input = tf.placeholder(tf.float32, shape=[self.minibatch, 100])
      self.labels = tf.placeholder(tf.float32, shape=[self.minibatch, 10])
      glorot_bounds = np.sqrt(3.)/np.sqrt(10.+10.)
      self.W = tf.Variable(tf.random_uniform(shape=[100,10], minval=-glorot_bounds, maxval=glorot_bounds), name='W')
      self.b = tf.Variable(tf.zeros(shape=[10]), name='b')
      self.inference = tf.nn.softmax( tf.add(tf.matmul(self.input, self.W, name='matmul'), self.b, name='sum'), name='inference' )

      self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits( logits=self.inference, labels=self.labels ), name='loss')
      self.train = tf.train.GradientDescentOptimizer(1.0).minimize(self.loss)

  def get_graph(self):
    return self.g

  def get_weights(self):
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

  def run_train(self, runstep=None, n_steps=1, *args, **kwargs):
    b0,b1 = 0,self.minibatch
    for step in range(0,n_steps):
      b0 = (b0+self.minibatch)%self.trainsize
      b1 = min( (b0+self.minibatch), self.trainsize)
      _,loss = runstep(self.session, fetches=[self.train, self.loss], feed_dict={self.input: self.example_data[b0:b1], self.labels: self.example_labels[b0:b1]})
    return loss

  def run_inference(self, runstep=None, n_steps=1, *args, **kwargs):
    b0,b1 = 0,self.minibatch
    for step in range(0,n_steps):
      b0 = (b0+self.minibatch)%self.trainsize
      b1 = min( (b0+self.minibatch), self.trainsize)
      inf = runstep(self.session, fetches=[self.inference], feed_dict={self.input: self.example_data[b0:b1]})
    return inf

  def get_activations(self, runstep=None, *args, **kwargs):
    # FIXME: left blank pending a definition of "activation" (c.f. core/model.py)
    return dict()

def __dnnamo_loader__():
  return SimpleNNet()
