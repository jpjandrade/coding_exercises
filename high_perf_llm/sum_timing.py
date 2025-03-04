import datetime
import jax

MATRIX_DIM = 32768
STEPS = 10

A = jax.numpy.ones((MATRIX_DIM, MATRIX_DIM))
B = jax.numpy.ones((MATRIX_DIM, MATRIX_DIM))
C = jax.numpy.zeros((MATRIX_DIM, MATRIX_DIM))

num_bytes = A.size * 4  # 4 bytes per float32

# Passing A, B from memory to chip and C back to memory.
total_num_bytes_crossing_hmb = num_bytes * 3

total_num_flops = MATRIX_DIM * MATRIX_DIM

jax.profiler.start_trace('/tmp/trace')
start_time = datetime.datetime.now()

for _ in range(STEPS):
    C = A + B + C


end_time = datetime.datetime.now()

average_time = (end_time - start_time).total_seconds() / STEPS

print(
    f"{average_time}, TFLOPS/s {total_num_flops / average_time / 1e12}, GB/s {total_num_bytes_crossing_hmb / average_time / 1e9}"
)
