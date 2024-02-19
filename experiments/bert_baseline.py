from transformers import AutoTokenizer, TFAutoModel
import tensorflow as tf
import pandas as pd
import numpy as np

MODEL_NAME = 'bert-base-uncased'
EPOCHS = 1
BATCH_SIZE = 32
MAX_LENGTH = 512
LEARNING_RATE = 1e-5
DROPOUT_RATE = 0.8
MODEL_OUT_FILE_NAME = 'bert_evidence_baseline_model.h5'
TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_data(questions, evidence):
    inputs = TOKENIZER(questions, evidence, padding='max_length', truncation=True, max_length=MAX_LENGTH, return_tensors='tf')
    return inputs

train_dataset = pd.read_csv('../data/emrqa_evidence_train.csv')
val_dataset = pd.read_csv('../data/emrqa_evidence_val.csv')

train_x = train_dataset.apply(lambda x: tokenize_data(x['question'], x['evidence']), axis=1).to_numpy()
train_y = np.array(train_dataset[['start_token', 'end_token']]).astype('float32')
val_x = val_dataset.apply(lambda x: tokenize_data(x['question'], x['evidence']), axis=1).to_numpy()
val_y = np.array(val_dataset[['start_token', 'end_token']]).astype('float32')

model = TFAutoModel.from_pretrained(MODEL_NAME)

optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_x, train_y, validation_data=(val_x, val_y), epochs=EPOCHS, batch_size=BATCH_SIZE)

