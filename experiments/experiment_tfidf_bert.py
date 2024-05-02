import pandas as pd
import numpy as np
from Models import BertBaseline
from transformers import AutoTokenizer

def train():
    print("Training Bert with TF-IDF Evidence")
    tokenizer = AutoTokenizer.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')
    train_dataset = pd.read_csv('../data/emrqa_tfidf_evidence_train.csv')
    val_dataset = pd.read_csv('../data/emrqa_tfidf_evidence_val.csv')

    # tokenize the training and validation datasets
    tokenized_train = tokenizer(train_dataset['question'].to_list(),
                                train_dataset['tf-idf_evidence'].to_list(),
                                padding='max_length',
                                truncation=True,
                                max_length=512,
                                return_tensors='tf')
    tokenized_val = tokenizer(val_dataset['question'].to_list(),
                              val_dataset['tf-idf_evidence'].to_list(),
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
    print("Saving Bert with TF-IDF Evidence")
    bert_baseline.model.save('bert_tfidf_model')

def test():
    model = BertBaseline()
    model.load_model_weights('../models/bert_evidence_baseline_weights.h5')
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    test_dataset = pd.read_csv('../data/emrqa_evidence_test.csv')
    tokenized_test = tokenizer(test_dataset['question'].to_list(),
                               test_dataset['tf-idf_evidence'].to_list(),
                               padding='max_length',
                               truncation=True,
                               max_length=512,
                               return_tensors='tf')
    test_x = (tokenized_test['input_ids'], tokenized_test['attention_mask'])
    test_y = np.stack((np.eye(512)[test_dataset['start_token']], np.eye(512)[test_dataset['end_token']]), axis=1)
    results = model.model.predict(test_x, batch_size=32)
    print(results.shape)


if __name__ == '__main__':
    train()



