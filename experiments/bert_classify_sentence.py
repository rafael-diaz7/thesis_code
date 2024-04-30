import pandas as pd
from Models import SentenceClassificationBert
from transformers import AutoTokenizer
import numpy as np
from tensorflow.keras.utils import Sequence
import math



# TODO change batch size to 32 IF it works for you, 16 doesn't work for me
class DataSequence(Sequence):
    def __init__(self, df, tokenizer, batch_size=16, max_length=512):
        self.df = df
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_length = max_length
        self.questions = df['question'].tolist()
        self.sentences = df['sentence'].tolist()
        self.answers = df['expected_evidence'].tolist()
        self.indices = np.arange(len(df))

    def __len__(self):
        return int(math.ceil(len(self.df) / self.batch_size))

    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size: (idx + 1) * self.batch_size]
        batch_questions = [self.questions[i] for i in batch_indices]
        batch_sentences = [self.sentences[i] for i in batch_indices]
        batch_y = [self.answers[i] for i in batch_indices]
        tokenized_batch = self.tokenizer(batch_questions, batch_sentences, truncation=True, padding='max_length',
                                         max_length=self.max_length, return_tensors='tf')
        return (tokenized_batch['input_ids'], tokenized_batch['attention_mask']), np.array(batch_y)

def train():
    print("Running Bert Sentence Classification Model")
    tokenizer = AutoTokenizer.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')
    train_dataset = pd.read_csv('../data/emrqa_sentence_train.csv')
    val_dataset = pd.read_csv('../data/emrqa_sentence_val.csv')
    print("Creating tokenized data")
    train_generator = DataSequence(train_dataset, tokenizer)
    val_generator = DataSequence(val_dataset, tokenizer)
    print("Creating model")
    bert = SentenceClassificationBert()
    print("Training Model")
    history = bert.train(train_generator, val_generator, epochs=50)
    print("Saving Bert Sentence Classification Model")
    bert.model.save('bert_sentence_classification_model')


if __name__ == '__main__':
    train()



