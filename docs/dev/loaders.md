Because there are many ways to implement a neural net, there's no one way to capture them.
Machine learning frameworks sometimes provide externalizeable formats, which expose a varying degree of functionality.
Other times, a network is built into a software library and can only be used directly.
To solve this, Dnnamo provides *loaders*.
Loaders are bits of code that take in the location or name of a network and return a corresponding Dnnamo model object.
Depending on the loader and the implementation of the model, this object might be one of several [model types](/dev/models/index.html#types-of-models).


