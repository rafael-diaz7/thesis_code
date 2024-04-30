import pandas as pd
from nltk import sent_tokenize

# We need to do the following here:
# 1. Remove all unnecessary columns, we only need Question, Answer, And Expected Evidence
# 2. Use nltk to tokenize the sentences and mark sentences that contain the answer as expected evidence
# 3. For every row, we make a new sample that is the question, the sentence, whether it contains the answer

train_df = pd.read_csv("../data/emrqa_evidence_train.csv")
test_df = pd.read_csv("../data/emrqa_evidence_test.csv")
val_df = pd.read_csv("../data/emrqa_evidence_val.csv")


def prepare_data(df):
    questions = []
    sentence_list = []
    expected_evidences = []
    for index, row in df.iterrows():
        question = row["question"]
        answer = row["answer"]
        context = row["context"]
        sentences = sent_tokenize(context)
        for sentence in sentences:
            if answer in sentence:
                expected_evidence = 1
            else:
                expected_evidence = 0
            questions.append(question)
            sentence_list.append(sentence)
            expected_evidences.append(expected_evidence)
    return pd.DataFrame({"question": questions, "sentence": sentence_list, "expected_evidence": expected_evidences})


prepare_data(train_df).to_csv("../data/emrqa_sentence_train.csv", index=False)
prepare_data(test_df).to_csv("../data/emrqa_sentence_test.csv", index=False)
prepare_data(val_df).to_csv("../data/emrqa_sentence_val.csv", index=False)

