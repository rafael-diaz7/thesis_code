"""
this script will only be used for data generation for the first step of TF-IDF -> BERT pipeline
i'm hoping to convert this to the end to end pipeline, but for now, this will only be data generation
just for ease of using it on dr. henry's computer

TODO: try the clinitokenizer instead of the nltk sentence tokenizer
^ it's specifically trained on i2b2 data, so it should be significantly better.
it can be hotswapped with sent_tokenize, so that's nice. install just didn't work for me
"""
import pandas as pd
from transformers import AutoTokenizer
from nltk.corpus import stopwords
from clinitokenizer.tokenize import clini_tokenize as sent_tokenize
from gensim.corpora import Dictionary
from gensim.models import TfidfModel
from gensim.matutils import cossim


tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
stopwords = set(stopwords.words('english'))

sentence_cache = {}

def create_tfidf_evidence(df, fp):
    all_context_texts = df['context'].unique().tolist()

    # clean stop words from the context
    cleaned_contexts = []
    for context in all_context_texts:
        cleaned_contexts.append([word for word in context.split() if word.lower() not in stopwords])
    dictionary = Dictionary(cleaned_contexts)
    corpus = [dictionary.doc2bow(context) for context in cleaned_contexts]
    tfidf = TfidfModel(corpus)

    df['tf-idf_evidence'] = df.apply(lambda x:
                                     apply_tfidf_to_context(x['question'], x['context'], dictionary, tfidf),
                                     axis=1)
    df = df[['question', 'tf-idf_evidence', 'answer']]
    df[['start_token', 'end_token']] = df.apply(
        lambda x: calc_token_spans(x['question'], x['tf-idf_evidence'], x['answer']), axis=1, result_type='expand')

    df.to_csv(fp, index=False)


def apply_tfidf_to_context(question, context, dictionary, tfidf):
    if context in sentence_cache:
        sentences = sentence_cache[context]
    else:
        sentences = sent_tokenize(context)
    # get vector for each sentence
    sentence_vectors = [tfidf[dictionary.doc2bow(sentence.split())] for sentence in sentences]
    question_vector = tfidf[dictionary.doc2bow(question.split())]
    cosine_similarities = [cossim(question_vector, sentence_vector) for sentence_vector in sentence_vectors]
    max_index = cosine_similarities.index(max(cosine_similarities))
    return sentences[max_index]

def calc_token_spans(question, evidence, answer):
    inputs = tokenizer(question, evidence, return_offsets_mapping=True)
    offset = inputs['offset_mapping']
    calc_answer_start = evidence.find(answer)
    end_char = calc_answer_start + len(answer)
    sequence_ids = inputs.sequence_ids()

    if calc_answer_start == -1:
        return 0, 0

    idx = 0
    while sequence_ids[idx] != 1:
        idx += 1
    context_start = idx
    while sequence_ids[idx] == 1:
        idx += 1
    context_end = idx - 1

    idx = context_start
    try:
        while idx <= context_end and offset[idx][0] <= calc_answer_start:
            idx += 1
        start_position = idx - 1

        idx = context_end
        while idx >= context_start and offset[idx][1] >= end_char:
            idx -= 1
        end_position = idx + 1
    except:
        print(offset)
        print(idx)
        print(len(offset))
        print(offset[idx])

    if start_position == end_position:
        return 0, 0
    if start_position >= 512 or end_position >= 512:
        return 0, 0
    return start_position, end_position


if __name__ == '__main__':
    df_train = pd.read_csv('../data/emrqa_context_train.csv')
    df_val = pd.read_csv('../data/emrqa_context_val.csv')
    df_test = pd.read_csv('../data/emrqa_context_test.csv')

    create_tfidf_evidence(df_train, '../data/emrqa_clini_tfidf_evidence_train.csv')
    create_tfidf_evidence(df_val, '../data/emrqa_clini_tfidf_evidence_val.csv')
    create_tfidf_evidence(df_test, '../data/emrqa_clini_tfidf_evidence_test.csv')
