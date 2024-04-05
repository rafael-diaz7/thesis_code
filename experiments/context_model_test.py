from transformers import AutoTokenizer
import pandas as pd
import numpy as np
from experiments.Models import BertBaseline

model = BertBaseline()
model.load_model_weights('../models/bert_context_baseline_weights.h5')
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
test_dataset = pd.read_csv('../data/emrqa_context_test.csv')
tokenized_test = tokenizer(test_dataset['question'].to_list(),
                           test_dataset['context'].to_list(),
                           padding='max_length',
                           truncation=True,
                           max_length=512,
                           return_tensors='tf')
test_x = (tokenized_test['input_ids'], tokenized_test['attention_mask'])
test_y = np.stack((np.eye(512)[test_dataset['start_token']], np.eye(512)[test_dataset['end_token']]), axis=1)
results = model.model.predict(test_x, batch_size=5)

print(results)