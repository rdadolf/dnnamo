
# Support functions for writing TensorFlow wrappers
# 
# These methods supply some of the common functionality that users will need
# when creating a Dnnamo model wrapper. These are not a replacement for a model
# wrapper.

# Convert runmetadata to a dictionary {native_op -> usecs}

# Convert runmetadata to a TF.graph
# XXX: This may or may not be possible.
#   If it is not possible, we'll need to revisit the @graph property of
# frameworks so that @graph() and get_rungraph() return similar datatypes.
