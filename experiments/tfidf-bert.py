import pandas as pd
from nltk.corpus import stopwords
# import nltk
# nltk.download('stopwords')
from gensim.corpora import Dictionary
from gensim.models import TfidfModel

# load all data
# get all contexts
# make tfidf model using context
# get the location of the two most interesting terms
# get the 10 tokens before and after the two most interesting terms and combine to use as evidence

stopwords = set(stopwords.words('english'))
df = pd.read_csv('../data/emrqa_context_test.csv')

all_context_texts = df['context'].unique().tolist()

# clean stop words from the context
cleaned_contexts = []
for context in all_context_texts:
    cleaned_contexts.append([word for word in context.split() if word.lower() not in stopwords])
dictionary = Dictionary(cleaned_contexts)
corpus = [dictionary.doc2bow(context) for context in cleaned_contexts]
tfidf = TfidfModel(corpus, smartirs='ntc')

def apply_tfidf_to_context(context):
    context = [word for word in context.split() if word.lower() not in stopwords]
    bow = dictionary.doc2bow(context)
    tfidf_weights = tfidf[bow]
    tfidf_weights = sorted(tfidf_weights, key=lambda x: x[1], reverse=True)
    term1, term2 = (dictionary[word[0]] for word in tfidf_weights[:2])

    # get the 10 words before and after the two most interesting terms
    term1_index = context.index(term1)
    term2_index = context.index(term2)
    return " ".join(context[term1_index - 10:term1_index + 11] + context[term2_index - 10:term2_index + 11])

df['tf-idf_evidence'] = df['context'].apply(apply_tfidf_to_context)
df = df[['question', 'tf-idf_evidence', 'answer']]