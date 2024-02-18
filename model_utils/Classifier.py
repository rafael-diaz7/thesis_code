from transformers import AutoTokenizer, TFAutoModel, TFBertModel
from keras.layers import *
from keras import Model
import tensorflow_addons as tfa
import os

from DataHandler import *
from CustomCallbacks import *
from Metrics import *
from abc import ABC, abstractmethod

class Classifier(ABC):
    '''
    Classifier class, which holds a language model and a classifier
    This class can be modified to create whatever architecture you want,
    however it requres the following instance variables:
    self.language_mode_name - this is a passed in variable and it specifies
       which HuggingFace language model to use
    self.tokenizer - this is created, but is just an instance of the tokenizer
       corresponding to the HuggingFace language model you use
    self.model - this is the Keras/Tensor flow classification model. This is
       what you can modify the architecture of, but it must be set

    Upon instantiation, the model is constructed. It should then be trained, and
    the model will then be saved and can be used for prediction.

    Training uses a DataHandler object, which inherits from a sequence object
    The DataHandler ensures that data is correctly divided into batches. 
    This could be done manually, but the DataHandler ensures it is done 
    correctly, and also allows for variable length batches, which massively
    increases the speed of training.
    '''
    BASEBERT = 'bert-base-uncased'

    # some default parameter values
    EPOCHS = 100
    BATCH_SIZE = 20
    MAX_LENGTH = 512
    # Note: MAX_LENGTH varies depending on the model. For Roberta, max_length = 768.
    #      For BERT its 512
    LEARNING_RATE = 1e-5 # This seems to be a pretty good default learning rate
    DROPOUT_RATE = 0.8
    LANGUAGE_MODEL_TRAINABLE = True
    MODEL_OUT_FILE_NAME = ''

    @abstractmethod
    def __init__(self, language_model_name, language_model_trainable=LANGUAGE_MODEL_TRAINABLE, max_length=MAX_LENGTH,
                 learning_rate=LEARNING_RATE, dropout_rate=DROPOUT_RATE):
        '''
        Initializer for a language model. This class should be extended, and
        the model should be built in the constructor. This constructor does
        nothing, since it is an abstract class. In the constructor however
        you must define:
        self.tokenizer 
        self.model
        '''
        self.tokenizer = None
        self.model = None
        self._language_model_name = language_model_name
        self._language_model_trainable = language_model_trainable
        self._max_length = max_length
        self._learning_rate = learning_rate
        self._dropout_rate = dropout_rate
        self.tokenizer = AutoTokenizer.from_pretrained(self._language_model_name)

    def load_language_model(self):

        # language_model = TFBertModel.from_pretrained('lm_weights_test_weights_out')

        # either load the language model locally or grab it from huggingface
        if os.path.isdir(self._language_model_name):
            language_model = TFBertModel.from_pretrained(self._language_model_name, from_pt=True)
            # else the language model can be grabbed directly from huggingface
        else:
            language_model = TFAutoModel.from_pretrained(self._language_model_name)

        # set properties
        language_model.trainable = self._language_model_trainable
        language_model.output_hidden_states = False

        # return the loaded model
        self.language_model = language_model
        return language_model

    def set_up_callbacks(self, early_stopping_monitor, early_stopping_mode, early_stopping_patience,
                         model_out_file_name, restore_best_weights, test_data):
        # set up callbacks
        callbacks = []
        if test_data is not None:
            if len(test_data) != 2:
                raise Exception("Error: test_data should be a tuple of (test_x, test_y)")
            callbacks.append(OutputTestSetPerformanceCallback(self, test_data[0], test_data[1]))
        if not model_out_file_name == '':
            callbacks.append(SaveModelWeightsCallback(self, model_out_file_name))
        if early_stopping_patience > 0:
            # try to correctly set the early stopping mode
            #  (checks if it should stop when increasing (max) or decreasing (min)
            if early_stopping_mode == '':
                if 'loss' in early_stopping_monitor.lower():
                    early_stopping_mode = 'min'
                elif 'f1' in early_stopping_monitor.lower():
                    early_stopping_mode = 'max'
                elif 'prec' in early_stopping_monitor.lower():
                    early_stopping_mode = 'max'
                elif 'rec' in early_stopping_monitor.lower():
                    early_stopping_mode = 'max'
                elif 'acc' in early_stopping_monitor.lower():
                    early_stopping_mode = 'max'
                else:
                    early_stopping_mode = 'auto'
                print("early_stopping_mode automatically set to " + str(early_stopping_mode))

            callbacks.append(tf.keras.callbacks.EarlyStopping(monitor=early_stopping_monitor, patience=early_stopping_patience,
                                           restore_best_weights=restore_best_weights, mode=early_stopping_mode))

        return callbacks

    def train(self, x, y, batch_size=BATCH_SIZE, validation_data=None, epochs=EPOCHS,
              model_out_file_name=MODEL_OUT_FILE_NAME, early_stopping_monitor='loss', early_stopping_patience=5,
              restore_best_weights=True, early_stopping_mode='', class_weights=None, test_data=None,
              training_data_handler=None, validation_data_handler=None):
        '''
        Trains the classifier
        :param x: the training data
        :param y: the training labels

        :param batch_size: the batch size
        :param: validation_data: a tuple containing x and y for a validation dataset
                so, validation_data[0] = val_x and validation_data[1] = val_y
                If validation data is passed in, then all metrics (including loss) will 
                report performance on the validation data
        :param: epochs: the number of epochs to train for
        '''

        # create the training data handler unless a special one was passed in
        if training_data_handler is None:
            # create a DataHAndler from the training data
            training_data_handler = TextClassificationDataHandler(x, y, batch_size, self)

        # create the validation data handler if there is validation data
        if validation_data is not None:
            if validation_data_handler is None:
                validation_data_handler = TextClassificationDataHandler(validation_data[0], validation_data[1],
                                                                        batch_size, self)

                # get the callbacks
        callbacks = self.set_up_callbacks(early_stopping_monitor, early_stopping_mode, early_stopping_patience,
                                          model_out_file_name, restore_best_weights, test_data)

        # fit the model to the training data
        self.model.fit(
            training_data_handler,
            epochs=epochs,
            validation_data=validation_data_handler,
            class_weight=class_weights,
            verbose=2,
            callbacks=callbacks
        )

    # function to predict using the NN
    def predict(self, x, batch_size=BATCH_SIZE):
        """
        Predicts labels for data
        :param x: data
        :return: predictions
        """
        if not isinstance(x, tf.keras.utils.Sequence):
            tokenized = self.tokenizer(x, padding=True, truncation=True, max_length=self._max_length,
                                       return_tensors='tf')
            x = (tokenized['input_ids'], tokenized['attention_mask'])

        return self.model.predict(x, batch_size=batch_size)

    # function to save the model weights
    def save_weights(self, filepath):
        """
        Saves the model weights
        :return: None
        """
        # if you want to just save the language model weights, you can
        # self.language_model.save_pretrained("lm_weights_"+filepath)

        # but, mostly we just want to save the entire model's weights
        self.model.save_weights(filepath)

    # function to load the model weights
    def load_weights(self, filepath):
        """
        Loads weights for the model
        The models are saved as three files:
            "checkpoint"
            "<model_name>.data-0000-of-00001" (maybe more of these)
            "<model_name>.index"
        :param filepath: the filepath and model_name (without extension) of the model
        :return: None
        """
        self.model.load_weights(filepath)

class BERTBaseline(Classifier):

    def __init__(self, language_model_name, num_classes, multi_class, language_model_trainable=Classifier.LANGUAGE_MODEL_TRAINABLE,
                 max_length=Classifier.MAX_LENGTH, learning_rate=Classifier.LEARNING_RATE,
                 dropout_rate=Classifier.DROPOUT_RATE):
        '''
        This Classifier is for token classification tasks, if it is a multi-label or binary task, set multi_class=false
        '''
        Classifier.__init__(self, language_model_name, language_model_trainable=language_model_trainable,
                            max_length=max_length, learning_rate=learning_rate, dropout_rate=dropout_rate)
        self._num_classes = num_classes
        self._multi_class = multi_class
        
        # create the language model
        language_model = self.load_language_model()

        # create the model
        # create the input layer, it contains the input ids (from tokenizer) and the
        # the padding mask (which masks padded values)
        input_ids = Input(shape=(None,), dtype=tf.int32, name="input_ids")
        input_padding_mask = Input(shape=(None,), dtype=tf.int32, name="input_padding_mask")

        # create the embeddings - the 0th index is the last hidden layer
        embeddings = language_model(input_ids=input_ids, attention_mask=input_padding_mask)[0]

        activation = 'sigmoid'
        loss_function = 'binary_crossentropy'

        output_layer = tf.keras.layers.Dense(self._num_classes, activation=activation)
        final_output = output_layer(embeddings)

        # set the metrics
        if self._multi_class:
            my_metrics = MultiClassTokenClassificationMetrics(self._num_classes)
        elif self._num_classes == 1:
            my_metrics = BinaryTokenClassificationMetrics(self._num_classes)
        else:
            my_metrics = MultiLabelTokenClassificationMetrics(self._num_classes)    
        metrics = my_metrics.get_all_metrics()

        # Combine the language model, set the optimizer and compile
        self.model = Model(inputs=[input_ids, input_padding_mask], outputs=[final_output])
        optimizer = tf.keras.optimizers.Adam(learning_rate=self._learning_rate)
        self.model.compile(
            optimizer=optimizer,
            loss=loss_function,
            metrics=metrics
        )

    def train(self, x, y, batch_size=Classifier.BATCH_SIZE, validation_data=None, epochs=Classifier.EPOCHS,
              model_out_file_name=Classifier.MODEL_OUT_FILE_NAME, early_stopping_monitor='loss',
              early_stopping_patience=5, restore_best_weights=True, early_stopping_mode='', class_weights=None,
              test_data=None, training_data_handler=None, validation_data_handler=None):
        '''
        Trains the classifier, just calls the Classifier.train, but sets up data handlers for token
        classification datasets rather than text classification datasets
        '''
        # create the training data handler unless a special one was passed in
        if training_data_handler is None:
            # create a DataHandler from the training data
            training_data_handler = TokenClassifierDataHandler(x, y, batch_size, self)

        # create the validation data handler if there is validation data
        if validation_data is not None:
            if validation_data_handler is None:
                validation_data_handler = TokenClassifierDataHandler(validation_data[0], validation_data[1], batch_size,
                                                                     self)

        # get the callbacks
        callbacks = self.set_up_callbacks(early_stopping_monitor, early_stopping_mode, early_stopping_patience,
                                          model_out_file_name, restore_best_weights, test_data)

        # fit the model to the training data
        self.model.fit(
            training_data_handler,
            epochs=epochs,
            validation_data=validation_data_handler,
            verbose=2,
            callbacks=callbacks
        )

    def predict(self, x, batch_size=Classifier.BATCH_SIZE):
        """
        Predicts labels for data
        :param x: data
        :param batch_size: batch size
        :return: predictions
        """
        if not isinstance(x, tf.keras.utils.Sequence):
            tokenized = self.tokenizer(list(x), padding=True, truncation=True, max_length=self._max_length,
                                       return_tensors='tf')
            x = (tokenized['input_ids'], tokenized['attention_mask'])
        return self.model.predict(x, batch_size=batch_size)

    def evaluate_predictions(self, pred_y, true_y, class_names):
        """
        Evaluates the predictions against true values. Predictions and Gold are from the classifier/dataset.
        They are a 3-D matrix [line, token, one-hot-vector of class]

        :param pred_y: matrix of predicted labels (one-hot encoded 0's and 1's)
        :param true_y: matrix of true values (one-hot encoded 0s and 1's)
        :param class_names: an ordered list of class names (strings)
        :param report_none: if True, then results for the none class will be reported and averaged into micro
                  and macro scores. A None class is automatically added to the class_names
        :param multi_class: indicates if the labels are multi-class or multi-label
        """
        # Set parameters depending on the classification type
        # multi-class_case
        if self._multi_class:
            report_zeroth = False
        # multi-label/binary case
        else:
            report_zeroth = True

        binary_classification = False
        if len(class_names) == 1:
            binary_classification = True
            pred_y = np.round(pred_y)

        # grab dimensions
        num_lines = pred_y.shape[0]
        padded_token_length = pred_y.shape[1]
        num_classes = pred_y.shape[2]

        # ensure the num predicted lines = num gold lines
        if num_lines != len(true_y):
            print("ERROR: the number of predicted lines does not equal the number of gold lines. "
                  f"\n  num predicted lines = {num_lines}, num gold_lines = {len(true_y)}"
                  "\n   Do your prediction and gold datasets match?")
            exit()

        # ensure the class_names length matches the number of predicted classes
        if num_classes != len(class_names):
            print("ERROR in evaluate_predictions: number of predicted classes not equal to the number of "
                  + "provided class names. Did you forget about a None class?")
            return

        # flatten the predictions. So, it is one prediction per token
        gold_flat = []
        pred_flat = []
        for i in range(num_lines):
            # get the gold and predictions for this line
            line_gold = true_y[i][:, :]
            line_pred = pred_y[i, :, :]

            # the gold contains the number of tokens (predictions are padded)
            # remove padded predictions
            num_tokens = line_gold.shape[0]
            line_pred = pred_y[i, :num_tokens, :]

            # convert token classifications to categorical.
            if self._multi_class:
                line_gold_categorical = np.argmax(line_gold, axis=1)
                line_pred_categorical = np.argmax(line_pred, axis=1)

            else: # multilabel or binary
                # Argmax returns 0 if everything is 0,
                # so, determine if classification is None class. If it's not, add 1 to the argmax
                not_none = np.max(line_gold, axis=1) > 0
                line_gold_categorical = np.argmax(line_gold, axis=1) + not_none
                not_none = np.max(line_pred, axis=1) > 0
                line_pred_categorical = np.argmax(line_pred, axis=1) + not_none
                #TODO - this doesn't make sense to convert to categorical for multi-label. But, it does for binary
                #TODO - for binary we need to round. I should probably do it here rather than passing it in first

            # add to the flattened list of labels
            gold_flat.extend(line_gold_categorical.tolist())
            pred_flat.extend(line_pred_categorical.tolist())

        # initialize the dictionaries
        num_classes = len(class_names)
        tp = []
        fp = []
        fn = []
        for i in range(num_classes):
            tp.append(0)
            fp.append(0)
            fn.append(0)

        # count the tps, fps, fns
        num_samples = len(pred_flat)
        for i in range(num_samples):

            # calculating tp, fp, fn for multiclass and binary is slightly different
            # TODO - this may not work for multilabel
            if not binary_classification:
                true_index = gold_flat[i]
                pred_index = pred_flat[i]
                correct = pred_flat[i] == gold_flat[i]
                if correct:
                    tp[true_index] += 1
                else:
                    fp[pred_index] += 1
                    fn[true_index] += 1

            if binary_classification:
                if gold_flat[i] == 1 and pred_flat[i] == 1:
                    tp[0] += 1
                elif gold_flat[i] == 0 and pred_flat[i] == 1:
                    fp[0] += 1
                elif gold_flat[i] == 1 and pred_flat[i] == 0:
                    fn[0] += 1
                #elif gold_flat[i] == 0 and pred_flat[i] == 0:
                    #tn += 1

        # convert tp, fp, fn into arrays and trim if not reporting none
        if report_zeroth: # report for all classes (binary and multi-label)
            tp = np.array(tp)
            fp = np.array(fp)
            fn = np.array(fn)
        else: # report for all but the None class (multiclass typically)
            # take [1:] to remove the None Class
            tp = np.array(tp)[1:]
            fp = np.array(fp)[1:]
            fn = np.array(fn)[1:]

            if self._multi_class:
                class_names = class_names[1:]

        # account for 0s which will result in division by 0
        tp = tp.astype(float) # convert to a float
        tp[tp == 0] += 1e-10
                
        # calculate precision, recall, and f1 for each class
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1_score = (2 * precision * recall) / (precision + recall)
        support = tp + fn

        # calculate micro and macro averages
        macro_precision = np.mean(precision)
        macro_recall = np.mean(recall)
        macro_f1 = np.mean(f1_score)
        all_tp = np.sum(tp)
        all_fp = np.sum(fp)
        all_fn = np.sum(fn)
        micro_precision = all_tp / (all_tp + all_fp)
        micro_recall = all_tp / (all_tp + all_fn)
        micro_f1 = (2 * micro_precision * micro_recall) / (micro_precision + micro_recall)

        #micro_averaged_stats = {'precision': micro_precision, 'recall': micro_recall, 'f1': micro_f1}
        #macro_avareged_stats = {'precision': macro_precision, 'recall': macro_recall, 'f1': macro_f1}

        # output the results in a nice format
        print("{:<12s} {:<12s} {:<10s} {:}    {:}".format("", "precision", "recall", "f1-score", "support"))
        for i in range(len(class_names)):
            print(f"{class_names[i]:<10s} {precision[i]:10.3f} {recall[i]:10.3f} {f1_score[i]:10.3f} {support[i]:10}")
        print()
        print(f"micro avg {micro_precision:10.3f} {micro_recall:10.3f} {micro_f1:10.3f}")
        print(f"macro avg {macro_precision:10.3f} {macro_recall:10.3f} {macro_f1:10.3f}")