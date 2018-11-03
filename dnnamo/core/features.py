import numpy as np

class Features(object):

  def __init__(self):
    self._args = None
    self._vals = None

  def append(self, op_arguments, measurement):
    np_op_args = np.array(op_arguments)
    np_meas = np.array(measurement)
    assert len(np_op_args.shape)==1, 'op_arguments must be a 1-dimensional array or list'
    assert np_meas.shape==(), 'measurement must be a scalar'
    if self._args is None:
      self._args = np.array([op_arguments])
      self._vals = np.array([measurement])
    else:
      assert np.array(op_arguments).shape[0]==self._args.shape[1]
      self._args = np.vstack( (self._args,np.array(op_arguments)) )
      self._vals = np.append(self._vals, measurement)
    return self

  def extend(self, op_arguments_list, measurement_list):
    np_op_args_list = np.array(op_arguments_list)
    np_meas_list = np.array(measurement_list)
    assert len(np_op_args_list.shape)==2, 'op_arguments_list must be a 2-dimensional array or list'
    assert len(np_meas_list.shape)==1, 'measurement_list must be a 1-dimensional array or list'
    assert np_op_args_list.shape[0]==np_meas_list.shape[0], 'op arguments and measurements must have the same number of entries.'
    if self._args is None:
      self._args = np_op_args_list
      self._vals = np_meas_list
    else:
      assert np_op_args_list.shape[1]==self._args.shape[1], 'op arguments do not have the same shape as existing op arguments in features.'
      self._args = np.vstack( (self._args, np_op_args_list) )
      self._vals = np.hstack( (self._vals, np_meas_list) )
    return self

  def concatenate(self, features):
    return self.extend(features.op_arguments, features.measurements)

  @property
  def op_arguments(self):
    return self._args

  @property
  def measurements(self):
    return self._vals

  @property
  def array(self):
    '''Return a single 2-dimensional numpy array with feature vectors as rows.

    Note that each feature vector includes the measurement as the last element.'''
    return np.hstack( (self._args, np.expand_dims(self._vals,1)) )

  def write(self, filename):
    with open(filename,'wb') as f:
      np.savez(f, op_arguments=self._args, measurements=self._vals)
      f.flush()

  def read(self, filename):
    with open(filename,'rb') as f:
      npz_filemap = np.load(f)
      try:
        self._args = npz_filemap['op_arguments']
        self._vals = npz_filemap['measurements']
      except KeyError:
        raise IOError('No op arguments or measurements found in file "'+str(filename)+'". Is this a Dnnamo Features file?')
