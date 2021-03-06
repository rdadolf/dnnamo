from collections import namedtuple

class Datatag(namedtuple('Datatag',['name','mode','scope','ops'])):
  '''Categories of data.

  These categories describe different types of data collected in a DataManager.
  Datatags are used to access caches as well as to invalidate them, such as
  when a transformation modifies a model and renders the cached data obsolete.'''

  # VALID CHOICES FOR DATATAGS
  # While we don't check for these, it's good to know that some
  # combinations of datatags aren't ever used. For instance, there's no
  # way you can collect static intermediate values, because they're
  # computed at runtime. So while it's possible to describe using a
  # datatag, no data will ever be associated with it.
  ######################################################################
  # name    modes scopes  ops
  # graph   all   all     all
  # weights all   static  native
  # timing  all   dynamic all
  # ivalues all   dynamic native
  ######################################################################

  _names = ['graph','weights','timing','ivalues']
  _modes = ['training','inference']
  _scopes = ['static','dynamic']
  _ops = ['native','primitive']

  def typecheck(self):
    '''A pass-through function that checks the values of all tuple fields.

    Fields must contain one of the allowed values.'''

    if self.name not in self._names:
      raise TypeError, 'Invalid name '+str(self.name)+' in '+str(self)+'. Must be one of: '+','.join(self._names)
    if self.mode not in self._modes:
      raise TypeError, 'Invalid mode '+str(self.mode)+' in '+str(self)+'. Must be one of: '+','.join(self._modes)
    if self.scope not in self._scopes:
      raise TypeError, 'Invalid scope '+str(self.scope)+' in '+str(self)+'. Must be one of: '+','.join(self._scopes)
    if self.ops not in self._ops:
      raise TypeError, 'Invalid ops '+str(self.ops)+' in '+str(self)+'. Must be one of: '+','.join(self._ops)

    return self

  def mask_typecheck(self):
    '''A pass-through function that checks the values of all tuple fields.

    Fields must contain one of the allowed values or a wildcard.'''

    def _is_wildcard(value):
      return ( (value is None) or (value=='none') or (value=='all') )

    if (self.name not in self._names) and not _is_wildcard(self.name):
      raise TypeError, 'Invalid name '+str(self.name)+' in '+str(self)+'. Must be None or one of: '+','.join(self._names)+',none,all'
    if (self.mode not in self._modes) and not _is_wildcard(self.mode):
      raise TypeError, 'Invalid mode '+str(self.mode)+' in '+str(self)+'. Must be None or one of: '+','.join(self._modes)+',none,all'
    if (self.scope not in self._scopes) and not _is_wildcard(self.scope):
      raise TypeError, 'Invalid scope '+str(self.scope)+' in '+str(self)+'. Must be None or one of: '+','.join(self._scopes)+',none,all'
    if (self.ops not in self._ops) and not _is_wildcard(self.ops):
      raise TypeError, 'Invalid ops '+str(self.ops)+' in '+str(self)+'. Must be None or one of: '+','.join(self._ops)+',none,all'

    return self

  def expand_mask(self):
    '''Iterates over all possible datatags specified by this mask.'''
    if self.name is None or self.name=='none':
      names = []
    elif self.name=='all':
      names = self._names
    else:
      names = [self.name]
    if self.mode is None or self.name=='none':
      modes = []
    elif self.mode=='all':
      modes = self._modes
    else:
      modes = [self.mode]
    if self.scope is None or self.name=='none':
      scopes = []
    elif self.scope=='all':
      scopes = self._scopes
    else:
      scopes = [self.scope]
    if self.ops is None or self.name=='none':
      ops = []
    elif self.ops=='all':
      ops = self._ops
    else:
      ops = [self.ops]

    for name in names:
      for mode in modes:
        for scope in scopes:
          for op in ops:
            yield Datatag(name,mode,scope,op)

  @classmethod
  def all(cls):
    '''Iterates over all possible datatags.'''
    return Datatag('all','all','all','all').expand_mask()


class DataManager(object):
  registry = {} # class => [Datatag, ...]

  @classmethod
  def register(cls, new_class):
    if new_class not in cls.registry:
      cls.registry[new_class] = new_class().invalidation_tags

  @classmethod
  def _deregister(cls, victim_class):
    # Deregistration is not a normal use case. Mostly just for test.
    del cls.registry[victim_class]

  def __init__(self):
    self._cache = {_:None for _ in Datatag.all()}

  def invalidate(self, tag):
    for exact_tag in tag.mask_typecheck().expand_mask():
      self._cache[exact_tag] = None

  def __getitem__(self, tag):
    return self._cache[tag.typecheck()]

  def __contains__(self, tag):
    return tag in self._cache

  def __setitem__(self, tag, value):
    tag.typecheck()
    if value is None:
      raise ValueError, 'Attempted to insert None into data manager tag "'+str(tag)+'". This probably means the corresponding data collector failed.'
    self._cache[tag] = value
