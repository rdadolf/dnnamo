# Primitive Operations

Dnnamo can create a simplified performance model of neural networks called an abstract graph.
This graph represents the dataflow dependences between the individual operations that make up a net.
Many operations, however, tend to have the same performance behavior, even if their mathematical meaning is quite different.
For instance, consider the two matrix expressions `A - B` and `A mod B`, for two large matrices `A` and `B`.
Substituting these two operations for one another in a real neural network would almost cretainly break it.
From a performance standpoint, however, the operations are quite similar: both perform simple binary arithmetic elementwise over two similarly shaped matrix and produce an output of the same size.
Dnnamo represents these classes of similar operations with an abstract operation called a *primitive operation*, or primop.

# Available Primops

### `Primop_dot`

Matrix products with reduction along a single dimension.
This includes vector dot products, regular matrix multiplication, and tensor dot products.

### `Primop_hadamard`

Hadamard operations compute a function over every element of its arguments.
This subsumes both unary operations (e.g., square root, negation) as well as binary operations (e.g., addition, multiplication).
The name stems from the [Hadamard product](https://en.wikipedia.org/wiki/Hadamard_product_(matrices)) and its related family of matrix operations.

### `Primop_undef`

When a Dnnamo framework cannot find any other translation rule for a given native operation, it emits a `Primop_undef`.
These ops behave in most ways like a zero-cost operation (`Primop_zero`), but unlike those ops, this is probably incorrect.
In most cases, the right solution is to write a new framework translation rule to cover the unknown native operation.

### `Primop_zero`

These are zero-cost operations. This includes bookkeeping operations that exist to allow other functionality (constants, for instance) as well as operations that end up being removed as part of optimization steps (e.g., identity or pass-through ops).
