from transformers import AutoTokenizer
import pandas as pd
from nltk import sent_tokenize
from tqdm import tqdm
import numpy as np
from experiments.Models import SentenceClassificationBert
tqdm.pandas()
tokenizer = AutoTokenizer.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')
model = SentenceClassificationBert()

# model.model.load_weights("bert_sentence_classification_model/variables/variables")
model.model.load_weights("../models/bert_sentence_classification_model")

def classify_df_iteration(df, context_tokenized_key):
    for ind, i in tqdm(df.iterrows()):
        sentences = context_tokenized_key[i["context"]]
        questions = [i['question']] * len(sentences)
        # tokenize and predict in batch
        tokenized = tokenizer(questions, sentences,
                           padding='max_length',
                           truncation=True,
                           max_length=512,
                           return_tensors='tf')
        test_x = (tokenized['input_ids'], tokenized['attention_mask'])
        predictions = (model.model.predict(test_x) >= 0.5).reshape(-1)
        correct = np.array(questions)[predictions]
        df.at[ind, "predicted_evidence"] = "".join(correct)

def classify_df_apply(row, context_tokenized_key):
    sentences = context_tokenized_key[row["context"]]
    questions = [row['question']] * len(sentences)
    # tokenize and predict in batch
    tokenized = tokenizer(questions, sentences,
                           padding='max_length',
                           truncation=True,
                           max_length=512,
                           return_tensors='tf')
    test_x = (tokenized['input_ids'], tokenized['attention_mask'])
    predictions = (model.model.predict(test_x, verbose=False) >= 0.5).reshape(-1)
    correct = np.array(sentences)[predictions]
    return "".join(correct)

def val_set_only():
    val_df = pd.read_csv("../data/emrqa_context_val.csv")
    context_tokenized_key = {context: sent_tokenize(context) for context in val_df["context"].unique().tolist()}
    val_df["predicted_evidence"] = val_df.progress_apply(lambda row: classify_df_apply(row, context_tokenized_key), axis=1)
    val_df.to_csv("../data/emrqa_context_val_predicted.csv", index=False)

def test_set_only():
    test_df = pd.read_csv("../data/emrqa_context_test.csv")
    context_tokenized_key = {context: sent_tokenize(context) for context in test_df["context"].unique().tolist()}
    test_df["predicted_evidence"] = test_df.progress_apply(lambda row: classify_df_apply(row, context_tokenized_key), axis=1)
    test_df.to_csv("../data/emrqa_context_test_predicted.csv", index=False)

def train_set_only():
    train_df = pd.read_csv("../data/emrqa_context_train.csv")
    context_tokenized_key = {context: sent_tokenize(context) for context in train_df["context"].unique().tolist()}
    train_df["predicted_evidence"] = train_df.progress_apply(lambda row: classify_df_apply(row, context_tokenized_key), axis=1)
    train_df.to_csv("../data/emrqa_context_train_predicted.csv", index=False)

def main():
    train_df = pd.read_csv("../data/emrqa_context_train.csv")
    test_df = pd.read_csv("../data/emrqa_context_test.csv")
    val_df = pd.read_csv("../data/emrqa_context_val.csv")

    context_tokenized_key = {context: sent_tokenize(context) for context in train_df["context"].unique().tolist() + test_df["context"].unique().tolist() + val_df["context"].unique().tolist()}

    train_df["predicted_evidence"] = train_df.progress_apply(lambda row: classify_df_apply(row, context_tokenized_key), axis=1)
    train_df.to_csv("../data/emrqa_context_train_predicted.csv", index=False)
    test_df["predicted_evidence"] = test_df.progress_apply(lambda row: classify_df_apply(row, context_tokenized_key), axis=1)
    test_df.to_csv("../data/emrqa_context_test_predicted.csv", index=False)
    val_df["predicted_evidence"] = val_df.progress_apply(lambda row: classify_df_apply(row, context_tokenized_key), axis=1)
    val_df.to_csv("../data/emrqa_context_val_predicted.csv", index=False)

def test():
    # mimic main function but on a significantly smaller dataset
    train_df = pd.read_csv("../data/emrqa_context_train.csv")
    test_df = pd.read_csv("../data/emrqa_context_test.csv")
    val_df = pd.read_csv("../data/emrqa_context_val.csv")

    context_tokenized_key = {context: sent_tokenize(context) for context in train_df["context"].unique().tolist() + test_df["context"].unique().tolist() + val_df["context"].unique().tolist()}

    train_df = train_df.head(10)
    test_df = test_df.head(10)
    val_df = val_df.head(10)

    train_df["predicted_evidence"] = train_df.progress_apply(lambda row: classify_df_apply(row, context_tokenized_key), axis=1)
    test_df["predicted_evidence"] = test_df.progress_apply(lambda row: classify_df_apply(row, context_tokenized_key), axis=1)
    val_df["predicted_evidence"] = val_df.progress_apply(lambda row: classify_df_apply(row, context_tokenized_key), axis=1)

    train_df.to_csv("../data/testtest.csv", index=False)

if __name__ == "__main__":
    train_set_only()
    # val_set_only()
    # test_set_only()