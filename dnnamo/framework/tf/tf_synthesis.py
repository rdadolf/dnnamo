import tensorflow as tf

from ...core.synthesis import SyntheticModel
from .tf_exemplar import TFSignatureType
from .tf_model import session_run, session_profile

class TFSyntheticModel(SyntheticModel):
  def __init__(self, exemplar):
    super(TFSyntheticModel,self).__init__(exemplar)
    self._g = tf.Graph()
    with self._g.as_default():
      self._sess = tf.Session()
    self._build_graph()

  def _build_graph(self):
    with self._g.as_default():
      self._inputs = [self._build_input(sig) for sig in self.exemplar.input_signature]
      self._outputs = self.exemplar.synthesize(self._inputs)

  def _build_input(self, input_sig):
    if isinstance(input_sig, TFSignatureType.Tensor):
      return tf.random_uniform(
        shape = input_sig.dims,
        minval = 0,
        maxval = 1,
        dtype = input_sig.dtype,
      )

  # Supported standard model interfaces

  def get_inference_graph(self):
    return self._g

  def run_inference(self, n_steps=1, *args, **kwargs):
    outs = []
    for _ in xrange(0,n_steps):
      out = session_run(self._sess, fetches=self._outputs)
      outs.append(out)
    return outs

  def profile_inference(self, n_steps=1, *args, **kwargs):
    rmds = []
    for _ in xrange(0,n_steps):
      rmd = session_profile(self._sess, fetches=self._outputs)
      rmds.append(rmd)
    return rmds

