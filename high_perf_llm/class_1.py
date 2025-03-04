import tensorflow as tf
import tensorflow_datasets as tfds

# Constants
SEQUENCE_LENGTH = 128
BATCH_IN_SEQUENCES = 256

dataset = tfds.load("lm1b", split="train[1shard]")
dataset = dataset.batch(256).prefetch(tf.data.AUTOTUNE)

for example in dataset.take(10):
    print(example)
    break
