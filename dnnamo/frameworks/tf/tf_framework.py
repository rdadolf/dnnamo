import sys
import os.path
import imp
import inspect
import random

import tensorflow as tf

from dnnamo.core.framework import Framework
from dnnamo.core.model import BaseModel, ImmutableModel, StaticModel, DynamicModel
from dnnamo.frameworks.tf.tf_translator import TFTranslator
from dnnamo.frameworks.tf.tf_runstep import _DefaultRunstep, _InstrumentedRunstep
from dnnamo.frameworks.tf.tf_stats import TFNativeStats

class TFFramework(Framework):
  def __init__(self, model=None):
    super(TFFramework, self).__init__(model)
    self._translator = TFTranslator()

  def _unique_module_name(self):
    # You can't imp.load_source() two modules with the same name. We want
    # want to import multiple modules rather often, and we almost never want
    # to import the same module twice. So we just choose an unused module name
    # and use that. We then promptly forget it, assuming we are willing to pay
    # the price to import a model twice.
    module_name = '_MODEL_'+str(random.randint(0,999999))
    while module_name in sys.modules:
      module_name = '_MODEL_'+str(random.randint(0,999999))
    return module_name

  def _parse_sourcename(self, source, modelname=None):
    # Models can be specified in four ways:
    # 1)  path/python-file.py
    # 2)  path/python-file.py:modelname
    # 3)  path/module-directory
    # 4)  path/module-directory:modelname
    # A modelname can also be asked for specifically by the framework.
    # This returns the separate components: (path,file-or-module,modelname)
    # The last element may be None, in which case the load function will
    # attempt to find a load the single Model in the file or module.
    if os.path.isfile(source) or ':' not in source:
      prefix,suffix = os.path.split( os.path.normpath(source) )
      return (prefix,suffix,modelname) # modelname could be None
    path,modelname2 = source.rsplit(':',1)
    if modelname is not None:
      assert modelname==modelname2, 'Multiple distinct modelname specified: "'+str(modelname)+'" vs "'+str(modelname2)+'"'
    modelname = modelname2
    prefix,suffix = os.path.split( os.path.normpath(path) )
    assert os.path.exists( path ), 'Cannot find model path: "'+str(path)+'"'
    return (prefix,suffix,modelname)

  def _is_class_a_model(self,typ):
    '''True if a *class* type is a subclass of BaseModel.

    Only works on the class type itself, not instances of that class.'''
    try:
      if issubclass(typ, BaseModel) and typ!=BaseModel:
        return True
    except TypeError:
      pass
    return False

  # pylint: disable=arguments-differ
  #   modelname, device, and init_options are TF-specific.
  #   other frameworks will have other special arguments.
  def load(self, source, modelname=None, device=None, init_options=None):
    (prefix,suffix,modelname) = self._parse_sourcename(source, modelname)
    source = os.path.join(prefix,suffix)

    print 'Decided on loading:',prefix,suffix,modelname

    # Load the source, depending on whether it's a file or module
    module_name = self._unique_module_name()
    if os.path.isfile(source):
      m = imp.load_source(module_name, source)
    elif os.path.isdir(source):
      # Abuse find_module to look only in this specific place.
      try:
        m = imp.load_module(module_name, *(imp.find_module(suffix,[prefix])))
      except ImportError as e:
        oldmsg = e.args
        newmsg = 'Error trying to import model ('+str(prefix)+','+str(suffix)+','+str(modelname)+'): '+str(oldmsg)
        e.args = tuple([newmsg] + list(e.args[1:]))
        raise
    else:
      raise IOError('No such file or directory: '+str(source))

    # Load and instantiate the model class
    model_pairs = dict(inspect.getmembers(m, self._is_class_a_model))
    if modelname is None: # No model specified, auto-detect a single model class
      assert len(model_pairs)>0, 'No models found in source "'+str(source)+'"'
      assert len(model_pairs)==1, 'Multiple models found in source "'+str(source)+'": '+','.join([str(m) for m in model_pairs])
      Modelclass = model_pairs.values()[0]
    else: # Model specified, instantiate it
      assert modelname in model_pairs, 'Model "'+str(modelname)+'" not found in source "'+str(source)+'"'
      Modelclass = model_pairs[modelname]

    self._model = Modelclass(device=device, init_options=init_options)

    return True

  def graph(self):
    if self._dgraph is None:
      assert self._model is not None, 'No model loaded. Run load(...) first.'
      self._dgraph = self._translator.translate( self._model )
    return self._dgraph

  def _transitive_closure(self, targets):
    # NOTE: Operational, but not currently used.
    #   Part of the problem is the need for a 'targets' argument, which is a
    #   model-specific runtime parameter. This not usually convenient to get
    #   when we would most want a static transitive closure function. I'm
    #   leaving it in here in case it is useful later.
    ops = set([])
    op_queue = []
    # Prime the queue
    for t in targets:
      if isinstance(t, tf.Tensor):
        #print 'adding new op',t.op.name
        op_queue.append(t.op)
        ops.add(t.op)
      else:
        #print 'adding new op',t.name
        op_queue.append(t)
        ops.add(t)
    # BFS the graph
    while len(op_queue)>0:
      op = op_queue[0]
      op_queue = op_queue[1:]
      for pre_op in [tensor.op for tensor in op.inputs]:
        if pre_op not in ops:
          #print 'adding new op',pre_op.name
          ops.add(pre_op)
          op_queue.append(pre_op)
      for pre_op in op.control_inputs:
        if pre_op not in ops:
          #print 'adding new control op',pre_op.name
          ops.add(pre_op)
          op_queue.append(pre_op)
    # All the ops necessary for computing the targets
    return list(ops)

  # This is a convenient shortcut so users don't have to go module surfing.
  DefaultRunstep = _DefaultRunstep
  InstrumentedRunstep = _InstrumentedRunstep

  def _build_native_stats(self, native_model, trace):
    return TFNativeStats(native_model, trace)
