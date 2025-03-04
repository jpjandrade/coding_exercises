import jax
import timing_util

MATRIX_DIM = 256


def f(A, B):
    return jax.nn.relu(A @ B)


jit_f = jax.jit(f)

for matrix_dim in [64, 128, 256, 512, 1024, 2048, 4096]:
    A = jax.numpy.ones((matrix_dim, matrix_dim))
    B = jax.numpy.ones((matrix_dim, matrix_dim))

    num_bytes = A.size * 4  # 4 bytes per float32
    total_num_bytes_crossing_to_chip = num_bytes * 3

    # N**3 times two because for every element in the result we do a multiplication and an addition.
    total_num_flops = 2 * matrix_dim * matrix_dim * matrix_dim + matrix_dim * matrix_dim

    average_time_sec_no_jit = timing_util.simple_timeit(f, A, B, task="f") / 1000

    average_time_sec_jit = timing_util.simple_timeit(jit_f, A, B, task="jit_f") / 1000

    print(f"arithmetic intensity {total_num_flops / total_num_bytes_crossing_to_chip}")

    print(
        f"No JIT: , {average_time_sec_no_jit} ,TFLOPS/s {total_num_flops / average_time_sec_no_jit / 1e12}, GB/s {total_num_bytes_crossing_to_chip / average_time_sec_no_jit / 1e9}"
    )

    print(
        f"JIT: , {average_time_sec_jit} ,TFLOPS/s {total_num_flops / average_time_sec_jit / 1e12}, GB/s {total_num_bytes_crossing_to_chip / average_time_sec_jit / 1e9}"
    )
