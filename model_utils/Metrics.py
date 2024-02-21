import tensorflow as tf
class ExtractiveQAMetrics():

    def convert_output_to_span(self, start_output, end_output):
        """
        Convert the output of the model to the span of the answer
        """
        start_index = tf.argmax(start_output, axis=1)
        end_index = tf.argmax(end_output, axis=1)
        # return tensor of start and end indices
        return tf.stack([start_index, end_index])

    def exact_match(self, y_true, y_pred):
        """
        Returns 1 if they are an exact match, 0 otherwise
        """
        print("y_pred shape", y_pred.shape)
        print(y_pred)
        print(y_pred.to_numpy())
        y_pred = self.convert_output_to_span(y_pred[0], y_pred[1])
        return tf.equal(tf.cast(y_true, dtype=tf.float32), y_pred)

    def count_shared_words(self, y_true, y_pred):
        """
        Returns the number of shared words between the y_true and the y_pred
        """
        y_pred = self.convert_output_to_span(y_pred[0], y_pred[1])
        print("y_true shape", y_true.shape)
        print(y_true)
        print(type(y_true))
        print(y_pred)
        print(type(y_pred))
        true_start, true_end = y_true
        pred_start, pred_end = tf.unstack(y_pred)

        start_max = tf.maximum(true_start, pred_start)
        end_min = tf.minimum(true_end, pred_end)

        shared_words = tf.maximum(0, end_min - start_max + 1)
        return shared_words

    def recall(self, y_true, y_pred):
        """
        Returns the precision of the y_true
        """
        return self.count_shared_words(y_true, y_pred) / (y_pred[1] - y_pred[0])

    def precision(self, y_true, y_pred):
        """
        Returns the recall of the y_true
        """
        return self.count_shared_words(y_true, y_pred) / (y_true[1] - y_true[0])

    def f1_score(self, y_true, y_pred):
        """
        Returns the f1 score of the y_true
        """
        return (2 * (self.precision(y_true, y_pred) * self.recall(y_true, y_pred))
                / (self.precision(y_true, y_pred) + self.recall(y_true, y_pred)))


if __name__ == "__main__":
    # Test the ExtractiveQAMetrics class
    y_true = tf.constant([[1, 2], [3, 4], [5, 6]])
    y_pred = tf.constant([[1, 3], [3, 4], [5, 6]])
    metrics = ExtractiveQAMetrics()
    print(metrics.exact_match(y_true, y_pred))
    print(metrics.count_shared_words(y_true, y_pred))
    print(metrics.recall(y_true, y_pred))
    print(metrics.precision(y_true, y_pred))
    print(metrics.f1_score(y_true, y_pred))