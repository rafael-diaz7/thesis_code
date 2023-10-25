from tensorflow.keras.callbacks import Callback, EarlyStopping
import numpy as np
import sklearn


class SaveModelWeightsCallback(Callback):
    "Saves the Model after each iteration of training"

    def __init__(self, classifier, weight_filename):
        Callback.__init__(self)
        self._classifier = classifier
        self._weight_filename = weight_filename

    def on_epoch_end(self, epoch, logs=None):
        self._classifier.save_weights(self._weight_filename)


class WriteMetrics(Callback):
    '''
    Example of a custom callback function. This callback prints information on
    epoch end, and does something (TODO, ask Rafael) when beginning training
    '''
    global mf

    def on_train_begin(self, logs=None):
        keys = list(logs.keys())
        print("At start; log keys: ".format(keys))
        print('GLOBAL FILE TEST:', mf)

    def on_epoch_end(self, epoch, logs=None):
        keys = list(logs.keys())
        print("End of epoch {}; log keys;: {}".format(epoch + 1, keys))
        print(list(logs.values()))
        vals = list(logs.values())
        print('GLOBAL TEST:', mf)
        with open(mf, 'a') as file:
            file.write("{},{:.4f},{:.4f},{:.4f},{:.4f}\n".format(epoch + 1, vals[0], vals[1], vals[2], vals[3]))


class OutputTestSetPerformanceCallback(Callback):
    '''
    Callback to output the test set performance to the console
    '''

    def __init__(self, classifier, test_x, test_y):
        Callback.__init__(self)
        self._classifier = classifier
        self._test_x = test_x
        self._test_y = test_y

    def on_epoch_end(self, epoch, logs=None):
        print("Epoch {} test set performance".format(epoch + 1))
        predictions = self._classifier.predict(self._test_x)
        predicted_labels = np.round(predictions)
        print(sklearn.metrics.classification_report(self._test_y, predicted_labels,
                                                    target_names=['TrIP', 'TrWP', 'TrCP', 'TrAP', 'TrNAP', 'TeRP',
                                                                  'TeCP', 'PIP']))
