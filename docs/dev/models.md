Neural networks can be implemented in a variety of ways, so applying standard tools to them can sometimes be tricky.
Dnnamo works around this issue by requiring models to be wrapped in a lightweight interface which exposes standard features.

# Types of Models

Not all neural networks are created equal, and sometimes it can be difficult to access the parts of a model.
For instance, a model integrated into a larger pipeline might be easy to export as a computational graph but difficult to run without the rest of the pieces.
Dnnamo provides several different interfaces to models to allow for these scenarios.
Not every model will implement every function, and while this will limit which tools can be applied to that model, some tools are better than no tools at all.

## Getting the graph

- `get_inference_graph()`: returns a framework-specific object corresponding to the underlying graph used for training. 
- `get_training_graph()`: returns a framework-specific object corresponding to the underlying graph used for training. 

Note that these two functions might return the same object.
Some frameworks use the same computational graph (in different ways) for both training and inference.

## Manipulating weights

**Warning: this section is under development and will likely change.**

- `get_weights(keys=None)`: returns a dictionary with all the model's weights, associated with a unique identifier. Optionally, you can pass in a list of keys. This will return a dictionary containing only the specified weights.
- `set_weights(kv)`: update the model's weights according to a dictionary of identifier-array pairs. Weight whose identifiers exist in the model but not in the supplied argument will not be modified.

## Running

These functions enable Dnnamo's tools to run a model and collect output.

- `run_inference(n_steps=1)`: runs the model on `n_steps` inputs and returns a list of corresponding model predictions.
- `run_training(n_steps=1)`: returns a list of tuples containing the loss and a "score" value for every step. The score is model-specific, typically some sort of accuracy number. `n_steps` is usually the number of minibatch steps taken, not the number of epochs.

## Profiling

Dnnamo needs access to performance information from the underlying machine learning framework.
Every framework has slightly different information, so both of these methods return framework-specific objects.

- `profile_training`: returns framework-specific profiling data. This data is interpreted by Dnnamo to provide a variety of analysis and optimization tools.
- `profile_inference`: returns framework-specific profiling data. This data is interpreted by Dnnamo to provide a variety of analysis and optimization tools.

## Intermediate values

**Warning: this section is under development and will likely change.**

Sometimes called "activations", intermediate values are the values that traverse dataflow edges in the computational graph of a neural network.
These values can be expensive to record, but some analysis and optimization tools require these values.

- `get_intermediates`: TBD

## `LearningModel`

