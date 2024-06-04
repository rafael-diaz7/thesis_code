import pandas as pd
from Models import BertBaseline, SentenceClassificationBert
import tensorflow as tf
import numpy as np
import re
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')

def replace_punctuation(s):
    s = re.sub(r'[^\w\s]', '', s)
    s = " ".join(s.split())
    return s

def exact_match(prediction, truth):
    return int(prediction.split() == truth.split())

def shared_words(prediction, truth):
    # find shared words between prediction and truth while
    a = set(prediction.split())
    b = set(truth.split())
    return a & b

def precision(prediction, truth):
    num = len(shared_words(prediction, truth))
    den = len(truth.split())
    if num == 0 or den == 0:
        return 0
    return num/den

def recall(prediction, truth):
    num = len(shared_words(prediction, truth))
    den = len(prediction.split())
    if num == 0 or den == 0:
        return 0
    return num/den

def f1_score(prediction, truth):
    p = precision(prediction, truth)
    r = recall(prediction, truth)
    if p + r == 0:
        return 0
    return 2*p*r/(p+r)

def compute_metrics(prediction, truth):
    prediction = replace_punctuation(prediction)
    truth = replace_punctuation(truth)
    return exact_match(prediction, truth), precision(prediction, truth), recall(prediction, truth), f1_score(prediction, truth)

def get_prediction(evidence, predicted_start, predicted_end):
    tokenized_evidence = tokenizer(evidence, padding=True, truncation=True, max_length=512)['input_ids']
    predicted_tokens = tokenized_evidence[predicted_start:predicted_end]
    return tokenizer.decode(predicted_tokens)

def get_bert_baseline_results():
    df = pd.read_csv('../data/emrqa_evidence_test.csv')
    model = BertBaseline()
    checkpoint = tf.train.Checkpoint(model=model.model)
    checkpoint.restore(tf.train.latest_checkpoint('../models/latest/bert_evidence_model'))
    # checkpoint.restore(tf.train.latest_checkpoint('bert_evidence_model'))
    tokenized_data = tokenizer(df['question'].to_list(),
                               df['evidence'].to_list(),
                               padding='max_length',
                               truncation=True,
                               max_length=512,
                               return_tensors='tf')
    predictions = model.model.predict((tokenized_data['input_ids'], tokenized_data['attention_mask']))
    starts = []
    ends = []
    for i in predictions:
        i = i.T
        starts.append(np.argmax(i[0]))
        ends.append(np.argmax(i[1]))
    predictions = pd.DataFrame({"start": starts, "end": ends})
    df['predicted_start'], df['predicted_end'] = starts, ends
    df['predictions'] = df.apply(
        lambda x: get_prediction(x['evidence'], x['predicted_start'], x['predicted_end']), axis=1)
    df['exact_match'], df['precision'], df['recall'], df['f1_score'] = zip(
        *df.apply(lambda x: compute_metrics(x['predictions'], x['answer']), axis=1))
    # get average metrics
    print(df[['exact_match', 'precision', 'recall', 'f1_score']].mean())

    # save results, results being the predictions and actual values
    save_df = df[['answer', 'predictions', 'exact_match', 'precision', 'recall', 'f1_score']]
    save_df.to_csv('../result_calculations/base_bert_ideal_results.csv', index=False)

def get_bert_context_results():
    df = pd.read_csv('../data/emrqa_context_test.csv')
    model = BertBaseline()
    checkpoint = tf.train.Checkpoint(model=model.model)
    checkpoint.restore(tf.train.latest_checkpoint('../models/latest/bert_context_model'))
    tokenized_data = tokenizer(df['question'].to_list(),
                               df['context'].to_list(),
                               padding='max_length',
                               truncation=True,
                               max_length=512,
                               return_tensors='tf')
    predictions = model.model.predict((tokenized_data['input_ids'], tokenized_data['attention_mask']))
    starts = []
    ends = []
    for i in predictions:
        i = i.T
        starts.append(np.argmax(i[0]))
        ends.append(np.argmax(i[1]))
    predictions = pd.DataFrame({"start": starts, "end": ends})
    df['predicted_start'], df['predicted_end'] = starts, ends
    df['predictions'] = df.apply(
        lambda x: get_prediction(x['context'], x['predicted_start'], x['predicted_end']), axis=1)
    df['exact_match'], df['precision'], df['recall'], df['f1_score'] = zip(
        *df.apply(lambda x: compute_metrics(x['predictions'], x['answer']), axis=1))
    # get average metrics
    print(df[['exact_match', 'precision', 'recall', 'f1_score']].mean())

    # save results, results being the predictions and actual values
    save_df = df[['answer', 'predictions', 'exact_match', 'precision', 'recall', 'f1_score']]
    save_df.to_csv('../result_calculations/bert_context_results.csv', index=False)

def get_tfidf_bert_results():
    df = pd.read_csv('../data/emrqa_tfidf_evidence_test.csv')
    model = BertBaseline()
    checkpoint = tf.train.Checkpoint(model=model.model)
    checkpoint.restore(tf.train.latest_checkpoint('../models/latest/bert_tfidf_model'))
    tokenized_data = tokenizer(df['question'].to_list(),
                               df['tf-idf_evidence'].to_list(),
                               padding='max_length',
                               truncation=True,
                               max_length=512,
                               return_tensors='tf')
    predictions = model.model.predict((tokenized_data['input_ids'], tokenized_data['attention_mask']))
    starts = []
    ends = []
    for i in predictions:
        i = i.T
        starts.append(np.argmax(i[0]))
        ends.append(np.argmax(i[1]))
    predictions = pd.DataFrame({"start": starts, "end": ends})
    df['predicted_start'], df['predicted_end'] = starts, ends
    df['predictions'] = df.apply(
        lambda x: get_prediction(x['tf-idf_evidence'], x['predicted_start'], x['predicted_end']), axis=1)
    df['exact_match'], df['precision'], df['recall'], df['f1_score'] = zip(
        *df.apply(lambda x: compute_metrics(x['predictions'], x['answer']), axis=1))
    # get average metrics
    print(df[['exact_match', 'precision', 'recall', 'f1_score']].mean())

    # save results, results being the predictions and actual values
    save_df = df[['answer', 'predictions', 'exact_match', 'precision', 'recall', 'f1_score']]
    save_df.to_csv('../result_calculations/tfidf_bert_results.csv', index=False)

def get_bert_to_bert_results():
    df = pd.read_csv('../data/emrqa_context_test_predicted.csv')
    model = BertBaseline()
    checkpoint = tf.train.Checkpoint(model=model.model)
    checkpoint.restore(tf.train.latest_checkpoint('../models/latest/bert_to_bert_model'))
    tokenized_data = tokenizer(df['question'].astype(str).to_list(),
                               df['predicted_evidence'].astype(str).to_list(),
                               padding='max_length',
                               truncation=True,
                               max_length=512,
                               return_tensors='tf')
    predictions = model.model.predict((tokenized_data['input_ids'], tokenized_data['attention_mask']))
    starts = []
    ends = []
    for i in predictions:
        i = i.T
        starts.append(np.argmax(i[0]))
        ends.append(np.argmax(i[1]))
    predictions = pd.DataFrame({"start": starts, "end": ends})
    df['predicted_start'], df['predicted_end'] = starts, ends
    df['predictions'] = df.apply(
        lambda x: get_prediction(str(x['predicted_evidence']), x['predicted_start'], x['predicted_end']), axis=1)
    df['exact_match'], df['precision'], df['recall'], df['f1_score'] = zip(
        *df.apply(lambda x: compute_metrics(x['predictions'], x['answer']), axis=1))
    # get average metrics
    print(df[['exact_match', 'precision', 'recall', 'f1_score']].mean())

    # save results, results being the predictions and actual values
    save_df = df[['answer', 'predictions', 'exact_match', 'precision', 'recall', 'f1_score']]
    save_df.to_csv('../result_calculations/bert_bert_results.csv', index=False)

def llm_evidence_results():
    df = pd.read_csv('../data/predicted/llama_evidence_predictions.csv')
    df['exact_match'], df['precision'], df['recall'], df['f1_score'] = zip(
        *df.apply(lambda x: compute_metrics(str(x['predicted']), str(x['true'])), axis=1))
    print(df[['exact_match', 'precision', 'recall', 'f1_score']].mean())
    save_df = df[['true', 'predicted', 'exact_match', 'precision', 'recall', 'f1_score']]
    save_df.to_csv('../result_calculations/llama_evidence_results.csv', index=False)

def llm_context_results():
    df = pd.read_csv('../data/predicted/llama_context_predictions.csv')
    df['exact_match'], df['precision'], df['recall'], df['f1_score'] = zip(
        *df.apply(lambda x: compute_metrics(str(x['predicted']), str(x['true'])), axis=1))
    print(df[['exact_match', 'precision', 'recall', 'f1_score']].mean())
    save_df = df[['true', 'predicted', 'exact_match', 'precision', 'recall', 'f1_score']]
    save_df.to_csv('../result_calculations/llama_context_results.csv', index=False)


if __name__ == "__main__":
    # llm_evidence_results()
    # llm_context_results()
    get_bert_baseline_results()
    # get_bert_context_results()
    # get_tfidf_bert_results()
    # get_bert_to_bert_results()