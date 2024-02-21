import tensorflow as tf

class AnswerSpan(tf.keras.layers.Layer):
    def call(self, inputs):
        start_logits, end_logits = inputs
        start_index = tf.argmax(start_logits, axis=1, output_type=tf.int32)
        end_index = tf.argmax(end_logits, axis=1, output_type=tf.int32)
        return start_index, end_index