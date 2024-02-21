from transformers import AutoTokenizer, TFAutoModel
from model_utils.Layers import AnswerSpan
from model_utils.Metrics import ExtractiveQAMetrics
import tensorflow as tf
import pandas as pd
import numpy as np

MODEL_NAME = 'bert-base-uncased'
EPOCHS = 1
BATCH_SIZE = 32

DROPOUT_RATE = 0.8
MODEL_OUT_FILE_NAME = 'bert_evidence_baseline_model.h5'
TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME)

class BertBaseline():
    def __init__(self):
        self.max_length = 512
        self.num_classes = 512
        self.language_model_name = 'bert-base-uncased'
        self.learning_rate = 1e-5
        tokenizer = AutoTokenizer.from_pretrained(self.language_model_name)
        self.tokenizer = tokenizer

        language_model = TFAutoModel.from_pretrained(self.language_model_name)

        input_ids = tf.keras.Input(shape=(None,), dtype=tf.int32, name='input_ids')
        attention_mask = tf.keras.Input(shape=(None,), dtype=tf.int32, name='attention_mask')

        embeddings = language_model(input_ids, attention_mask=attention_mask)[0]

        # output layer which will be a probability distribution over all the tokens indicating start or end of the answer
        start_logits = tf.keras.layers.Dense(1, name='start_logits', use_bias=False)(embeddings)
        end_logits = tf.keras.layers.Dense(1, name='end_logits', use_bias=False)(embeddings)

        answer_span_layer = AnswerSpan()
        # convert start logit to span of the answer using the most probable start index
        start_span, end_span = answer_span_layer([start_logits, end_logits])


        # concatenate into a single 2x1 tensor
        output = tf.keras.layers.Concatenate(name='output')([start_span, end_span])


        self.model = tf.keras.Model(inputs=[input_ids, attention_mask], outputs=output)
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        # metrics = []
        metrics = [ExtractiveQAMetrics().exact_match, ExtractiveQAMetrics().f1_score,
                   ExtractiveQAMetrics().recall, ExtractiveQAMetrics().precision]
        self.model.compile(optimizer=optimizer, loss=self.custom_loss, metrics=metrics)
        self.model.summary()

    def custom_loss(self, y_true, y_pred):
        start_loss = tf.keras.losses.sparse_categorical_crossentropy(
            tf.one_hot(tf.cast(y_true[:, 0], tf.int32), depth=self.num_classes), y_pred[0], from_logits=True)
        end_loss = tf.keras.losses.sparse_categorical_crossentropy(
            tf.one_hot(tf.cast(y_true[:, 1], tf.int32), depth=self.num_classes), y_pred[1], from_logits=True)
        return start_loss + end_loss


    def train(self, x_train, y_train, x_val, y_val, epochs=EPOCHS, batch_size=BATCH_SIZE):
            callbacks = []
            callbacks.append(tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=3,
                restore_best_weights=True,
                mode='min'))

            return self.model.fit(x_train,
                           y_train,
                           validation_data=(x_val, y_val),
                           epochs=epochs, batch_size=batch_size,
                           callbacks=callbacks,
                           verbose=1)

    def predict(self, x, batch_size=100):
        tokenized = self.tokenizer(x, padding='max_length', truncation=True, max_length=self.max_length, return_tensors='tf')
        x = (tokenized['input_ids'], tokenized['attention_mask'])

        start_probs, end_probs = self.model.predict(x, batch_size=batch_size)

        # Get the index with the highest probability for each position
        start_index = tf.argmax(start_probs, axis=1)
        end_index = tf.argmax(end_probs, axis=1)

        return start_index, end_index