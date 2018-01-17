import numpy as np

class Profile(object):
  def __init__(self):
    self._profile = {} # op -> [usecs, ...]
    self._steps = 0

  def __getitem__(self, key):
    return self._profile[key]

  def __setitem__(self, key, value):
    self._profile[key] = value

  def __len__(self):
    return len(self._profile)

  def items(self):
    return self._profile.items()

  def add(self, key, value):
    if key not in self._profile:
      self._profile[key] = []
    self._profile[key].append(value)
    self._steps = max(len(self._profile[key]), self._steps)

  @property
  def consistent(self):
    # Returns true if every op in the profile has the same number of data points.
    steps = [len(v) for v in self._profile.values()]
    return min(steps)==max(steps)

  def aggregate(self, mode):
    valid_modes = ['mean', 'first', 'last']
    # NOTE: if you implement new aggregation modes which select a single value
    # from the list, make sure you select the *same* value from every key/value
    # pair. If you cherry-pick (for instance, just applying "max" on every entry)
    # then the aggregate profile you return will contain imaginary data which
    # could be *very* wrong. (Imagine running 10 steps and taking an OS
    # interrupt on 10 different ops for each step. max'ing these profiles will
    # return an aggregate where each of those 10 ops is experiencing an OS
    # interrupt. This does not correspond to a measurement that ever existed.
    # Maybe this could be useful in niche cases, but be careful.
    if mode not in valid_modes:
      raise KeyError, 'Invalid aggregation mode '+str(mode)+', must be one of: '+', '.join(valid_modes)

    if not self.consistent:
      raise ValueError, 'Profile data is invalid. Some ops have more data than others, so it is not possible to create an accurate aggregate profile.'

    if mode=='mean':
      return {k:np.mean(v) for k,v in self._profile.items()}
    elif mode=='first':
      return {k:v[0] for k,v in self._profile.items()}
    elif mode=='last':
      return {k:v[-1] for k,v in self._profile.items()}
    return KeyError,'Invalid aggregation mode '+str(mode)
