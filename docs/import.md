NNModel is intended to be minimally invasive with respect to existing neural networks, but different people write code in different ways.
In order to allow NNModel to interact with user-supplied networks, it needs a couple of hooks, which are provided by wrapping the network in a simple Python object.

## The Model Interface

A model is either a python file or module which contains a subclass of `nnmodel.core.model.Model`.
To make things easier, each framework provides a `[Framework]Model` class which can be inherited directly.
Some frameworks will have functionality built in to these classes which will simplify the process.

While each framework will have slightly different implementations, the interface is the same.
There are five functions:

 - `__init__(device=None, init_options=None)`: This is a Python class, so a constructor is necessary. If you write your own constructor, you should call the framework-specific parent constructor before your custom code (i.e., with `super(MyClass).__init__()`). More details are below.

 - `model()`: This should return a reference to the native representation of the neural network. The exact representation may differ from framework to framework, but it should reflect a complete description of the model being run and measured. This function is *not* guaranteed to be called before running the model, and `setup` is *not* guaranteed to be called before running this function.

 - `setup(setup_options=None)`: Prepare the model for running. The framework will always call this function before running the model, but it need not (and, to avoid unnecessary computation, probably will not) call it before calling `model`. `setup_options` is a framework-specific argument to configure the model or runtime system. This argument must be a dictionary or `None`. Options may be ignored by the model, but it *must not* throw an exception on an unrecognized option.

 - `run(runstep=None, steps=1, *args, **kwargs)`: Execute the model. This is explained in greater detail below.

 - `teardown()`: Clean up after a model run. The framework will always call this function if it has called `setup`. If left unimplemented, an empty one will be inherited.

As much as possible, `__init__` should create a static representation of the graph and `setup` should handle everything required for running the model.
This is largely to make instantiating and statically analyzing a model (through the `model` function) cheap and fast.

### The `__init__` Method

The constructor is responsible for building (at least) the static components of the model.
While this function is optional, some features will not function without a conforming implementation.
This function takes two optional arguments, but it is recommended that the model writer honor both.

The `device` argument specifies the device the user would like to instantiate the model on. This is either a framework-specific string or `None`. If the value is `None`, no device has been chosen, and the model should choose an appropriate default. If the value is a string, the model should make its best effort to use that device. Hybrid (multi-device) scheduling is allowed, but if the device does not exist or the model knows the scheduling will be nonsensical from a performance standpoint, the model should prefer to fail with an error message.

The `options` argument allows the framework to pass in additional, model- or framework-specific parameters which are needed at construction time.
This argument must take a dictionary, which maps strings to arbitrary objects.
Each entry may be handled or ignored by the model, but it *must not* throw an exception if an argument is not recognized or supported.
Throwing an exception on improper values to a supported option is fine.

### The `run` Method

Many neural networks have rather complicated main loops, from which control is not easily relinquished.
Contrariwise, NNModel needs to have the ability to modify and control the way a network is run.
The comprimise is a callback mechanism.
A model writer must wrap their main loop in a `run` function, and they must modify it in two ways:

 1. Replace the innermost step with a call to the supplied `runstep` callback.
 2. Add a mechanism to halt the run after a fixed number of calls to `runstep`.

The `runstep` function is provided by the framework, and it executes a single step of the model (for many networks, this is equivalent to a single minibatch).
This callback exists because NNModel must be able to run the model in a couple different ways, so different `runstep` functions are supplied dynamically for different execution modes.

## A Sample Model Wrapper
