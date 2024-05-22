import pandas as pd
import numpy as np
from Models import BertBaseline
from transformers import AutoTokenizer

def train():
    print("Running Bert with Bert Predicted Evidence")
    tokenizer = AutoTokenizer.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')
    train_dataset = pd.read_csv('../data/emrqa_train_bert_predicted.csv')
    val_dataset = pd.read_csv('../data/emrqa_val_bert_predicted.csv')

    # tokenize the training and validation datasets
    tokenized_train = tokenizer(train_dataset['question'].astype(str).to_list(),
                                train_dataset['predicted_evidence'].astype(str).to_list(),
                                padding='max_length',
                                truncation=True,
                                max_length=512,
                                return_tensors='tf')
    tokenized_val = tokenizer(val_dataset['question'].astype(str).to_list(),
                              val_dataset['predicted_evidence'].astype(str).to_list(),
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
    print("Saving Bert with Bert Predicted Evidence")
    bert_baseline.model.save('bert_to_bert_model')


if __name__ == '__main__':
    train()



