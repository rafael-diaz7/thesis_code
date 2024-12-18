from transformers import AutoTokenizer, TFAutoModel
import tensorflow as tf
import numpy as np


MODEL_NAME = 'emilyalsentzer/Bio_ClinicalBERT'
EPOCHS = 50
BATCH_SIZE = 32

DROPOUT_RATE = 0.8
MODEL_OUT_FILE_NAME = 'bert_model'
TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME)

class BertBaseline():
    def __init__(self):
        self.max_length = 512
        self.num_classes = 512
        # use clinical bert
        self.language_model_name = 'emilyalsentzer/Bio_ClinicalBERT'
        self.learning_rate = 1e-5
        tokenizer = AutoTokenizer.from_pretrained(self.language_model_name)
        self.tokenizer = tokenizer

        language_model = TFAutoModel.from_pretrained(self.language_model_name)

        input_ids = tf.keras.Input(shape=(None,), dtype=tf.int32, name='input_ids')
        attention_mask = tf.keras.Input(shape=(None,), dtype=tf.int32, name='attention_mask')

        embeddings = language_model(input_ids, attention_mask=attention_mask)[0]

        spans = tf.keras.layers.Dense(2, activation='sigmoid')(embeddings)

        self.model = tf.keras.Model(inputs=[input_ids, attention_mask], outputs=spans)
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        metrics = []
        # metrics = [ExtractiveQAMetrics().exact_match, ExtractiveQAMetrics().f1_score,
        #            ExtractiveQAMetrics().recall, ExtractiveQAMetrics().precision]
        # metrics = [ExtractiveQAMetrics().precision]
        self.model.compile(optimizer=optimizer, loss=self.custom_loss, metrics=metrics)
        self.model.summary()


    def custom_loss(self, y_true, y_pred):
        y_pred_transposed = tf.transpose(y_pred, perm=[0, 2, 1])
        return tf.keras.losses.categorical_crossentropy(y_true, y_pred_transposed, from_logits=False)


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

    def save_model(self, filepath):
        self.model.save(filepath)
        self.model.save_weights(filepath)

    def load_model_weights(self, filepath):
        self.model.load_weights(filepath)


class SentenceClassificationBert():
    def __init__(self):
        self.max_length = 512
        self.num_classes = 512
        # use clinical bert
        self.language_model_name = 'emilyalsentzer/Bio_ClinicalBERT'
        self.learning_rate = 1e-5
        tokenizer = AutoTokenizer.from_pretrained(self.language_model_name)
        self.tokenizer = tokenizer

        language_model = TFAutoModel.from_pretrained(self.language_model_name)

        input_ids = tf.keras.Input(shape=(None,), dtype=tf.int32, name='input_ids')
        attention_mask = tf.keras.Input(shape=(None,), dtype=tf.int32, name='attention_mask')

        embeddings = language_model(input_ids, attention_mask=attention_mask)[0]

        sentence_representation_language_model = embeddings[:, 0, :]

        predicted = tf.keras.layers.Dense(1, activation='sigmoid')(sentence_representation_language_model)

        self.model = tf.keras.Model(inputs=[input_ids, attention_mask], outputs=predicted)
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        metrics = ['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
        self.model.compile(optimizer=optimizer, loss=tf.keras.losses.BinaryCrossentropy(), metrics=metrics)
        self.model.summary()

    def train(self, train_gen, val_gen, epochs=EPOCHS):
        callbacks = []
        callbacks.append(tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True,
            mode='min'))

        return self.model.fit(train_gen,
                              validation_data=val_gen,
                              epochs=epochs,
                              steps_per_epoch=len(train_gen),
                              validation_steps=len(val_gen),
                              callbacks=callbacks,
                              verbose=1)
