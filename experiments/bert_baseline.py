import pandas as pd
import numpy as np
from model_utils.Models import BertBaseline
from transformers import AutoTokenizer


if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    train_dataset = pd.read_csv('../data/emrqa_evidence_train.csv')
    val_dataset = pd.read_csv('../data/emrqa_evidence_val.csv')

    # tokenize the training and validation datasets
    tokenized_train = tokenizer(train_dataset['question'].to_list(),
                                train_dataset['evidence'].to_list(),
                                padding='max_length',
                                truncation=True,
                                max_length=512,
                                return_tensors='tf')
    tokenized_val = tokenizer(val_dataset['question'].to_list(),
                              val_dataset['evidence'].to_list(),
                              padding='max_length',
                              truncation=True,
                              max_length=512,
                              return_tensors='tf')

    train_x = (tokenized_train['input_ids'], tokenized_train['attention_mask'])
    # convert the start and end tokens to a one-hot encoded tensor and make it a 512x2 tensor
    train_y = np.stack((np.eye(512)[train_dataset['start_token']], np.eye(512)[train_dataset['end_token']]), axis=1)
    val_x = (tokenized_val['input_ids'], tokenized_val['attention_mask'])
    val_y = np.stack((np.eye(512)[val_dataset['start_token']], np.eye(512)[val_dataset['end_token']]), axis=1)
    bert_baseline = BertBaseline()
    history = bert_baseline.train(train_x, train_y, val_x, val_y, batch_size=32, epochs=50)
    bert_baseline.model.save('bert_evidence_baseline_model.h5')
    bert_baseline.model.save_weights('bert_evidence_baseline_weights.h5')


