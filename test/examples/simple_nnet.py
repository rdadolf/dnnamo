from dnnamo.frameworks.tf import TFModel

import tensorflow as tf
import numpy as np

class NativeOpSampleModel0(TFModel):
  def __init__(self, device=None, init_options=None):
    super(NativeOpSampleModel0,self).__init__(device=device, init_options=init_options)
    self.session = None
    self.g = tf.Graph()
    self.minibatch = 32
    self.trainsize = 2048
    with self.g.device(device):
      self._create_graph()

  def _create_graph(self):
    # 100:10 NN
    with self.g.as_default():
      self.input = tf.placeholder(tf.float32, shape=[self.minibatch, 100])
      self.labels = tf.placeholder(tf.float32, shape=[self.minibatch, 10])
      glorot_bounds = np.sqrt(3.)/np.sqrt(10.+10.)
      W = tf.Variable(tf.random_uniform(shape=[100,10], minval=-glorot_bounds, maxval=glorot_bounds), name='W')
      b = tf.Variable(tf.zeros(shape=[10]), name='b')
      self.inference = tf.nn.softmax( tf.add(tf.matmul(self.input,W, name='matmul'), b, name='sum'), name='inference' )

      self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits( self.inference, self.labels ), name='loss')
      self.train = tf.train.GradientDescentOptimizer(1.0).minimize(self.loss)

  def model(self):
    return self.g
    
  def setup(self, setup_options=None):
    super(NativeOpSampleModel0,self).setup(setup_options=setup_options)
    with self.g.as_default():
      if setup_options is not None:
        self.session = tf.Session(config=tf.ConfigProto(**setup_options))
      else:
        self.session = tf.Session()
      self.session.run(tf.global_variables_initializer())
      # Random training data
      self.example_data = np.random.rand(self.trainsize,100)
      self.example_labels = np.random.rand(self.trainsize,10)

  def run(self, runstep=None, n_steps=1, *args, **kwargs):
    b0,b1 = 0,self.minibatch
    for step in range(0,n_steps):
      b0 = (b0+self.minibatch)%self.trainsize
      b1 = min( (b0+self.minibatch), self.trainsize)
      _,loss = runstep(self.session, fetches=[self.train, self.loss], feed_dict={self.input: self.example_data[b0:b1], self.labels: self.example_labels[b0:b1]})

