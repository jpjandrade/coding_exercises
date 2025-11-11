import numpy as np
import jax


MATRIX_SIZE = 1024

A = jax.numpy.ones((MATRIX_SIZE, MATRIX_SIZE))

mesh = jax.sharding.Mesh(jax.devices(), "my_axis")
sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("my_axis"))
sharded_A = jax.device_put(A, sharding)

jax.debug.visualize_array_sharding(sharded_A)