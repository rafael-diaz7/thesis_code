import tensorflow as tf

print(tf.config.list_physical_devices())
print(f"GPUs Available: {len(tf.config.list_physical_devices('GPU'))>0}")