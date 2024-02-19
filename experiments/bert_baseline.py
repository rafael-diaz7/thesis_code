from transformers import AutoTokenizer, TFAutoModel
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

        self.model = tf.keras.Model(inputs=[input_ids, attention_mask], outputs=[start_logits, end_logits])
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.model.compile(optimizer=optimizer, loss=self.custom_loss, metrics=['accuracy'])
        print(self.model.summary())

    def custom_loss(self, y_true, y_pred):
        start_loss = tf.keras.losses.sparse_categorical_crossentropy(
            tf.one_hot(tf.cast(y_true[:, 0], tf.int32), depth=self.num_classes), y_pred[0], from_logits=True)
        end_loss = tf.keras.losses.sparse_categorical_crossentropy(
            tf.one_hot(tf.cast(y_true[:, 1], tf.int32), depth=self.num_classes), y_pred[1], from_logits=True)
        return start_loss + end_loss


    def train(self, x_train, y_train, x_val, y_val, epochs, batch_size):
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
                           callbacks=callbacks)

    def predict(self, x, batch_size=100):
        tokenized = self.tokenizer(x, padding='max_length', truncation=True, max_length=self.max_length, return_tensors='tf')
        x = (tokenized['input_ids'], tokenized['attention_mask'])

        start_probs, end_probs = self.model.predict(x, batch_size=batch_size)

        # Get the index with the highest probability for each position
        start_index = tf.argmax(start_probs, axis=1)
        end_index = tf.argmax(end_probs, axis=1)

        return start_index, end_index




if __name__ == '__main__':
    train_dataset = pd.read_csv('../data/emrqa_evidence_train.csv')
    val_dataset = pd.read_csv('../data/emrqa_evidence_val.csv')

    # tokenize the training and validation datasets
    tokenized_train = TOKENIZER(train_dataset['question'].to_list(), train_dataset['evidence'].to_list(), padding='max_length',
                                truncation=True, max_length=512, return_tensors='tf')
    tokenized_val = TOKENIZER(val_dataset['question'].to_list(), val_dataset['evidence'].to_list(), padding='max_length',
                              truncation=True, max_length=512, return_tensors='tf')
    train_x = (tokenized_train['input_ids'], tokenized_train['attention_mask'])
    train_y = np.stack((train_dataset['start_token'], train_dataset['end_token']), axis=1).astype('float32')
    val_x = (tokenized_val['input_ids'], tokenized_val['attention_mask'])
    val_y = np.stack((val_dataset['start_token'], val_dataset['end_token']), axis=1).astype('float32')
    bert_baseline = BertBaseline()
    history = bert_baseline.train(train_x, train_y, val_x, val_y, EPOCHS, BATCH_SIZE)


