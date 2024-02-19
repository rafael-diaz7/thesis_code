import json
import pandas as pd

question_all = []
context_all = []
answer_all = []
evidence_all = []

with open('../data/data.json') as f:
    for dataset in json.load(f)['data']:
        if dataset['title'] in ['obesity', 'smoking']:
            # ignoring these datasets because they are poorly formatted for extractive QA
            continue
        for paragraph in dataset['paragraphs']:
            context = "".join(paragraph['context'])
            for qa_pair in paragraph['qas']:
                questions = list(set(qa_pair['question']))
                answer = qa_pair['answers'][0]
                if answer['text'] == "":
                    # some items have no answer, likely due to it being one of the invalid datasets, but we should be
                    # cautious and check for this
                    continue
                if type(answer['text']) == list:
                    answer['text'] = answer['text'][0]
                if type(answer['evidence']) == list:
                    answer['evidence'] = answer['evidence'][0]
                question_all.append(questions[0].lower())
                context_all.append(context.lower())
                answer_all.append(answer['text'].lower())
                evidence_all.append(answer['evidence'].lower())

master_df = pd.DataFrame(
    {
        'question': question_all,
        'context': context_all,
        'answer': answer_all,
        'evidence': evidence_all,
    }
)
master_df = master_df.dropna()
master_df = master_df.sample(frac=1).reset_index(drop=True)  # shuffling the dataset
# 80/20 split for train/test
test_df = master_df.iloc[int(len(master_df)*0.8):]
train_val_df = master_df.iloc[:int(len(master_df)*0.8)]
# of the 80/20, we are doing a 90/10 split on the 80% for train/val
val_df = train_val_df.iloc[int(len(train_val_df)*0.9):]
train_df = train_val_df.iloc[:int(len(train_val_df)*0.9)]

train_df.to_csv('../data/emrqa_train.csv', index=False)
val_df.to_csv('../data/emrqa_val.csv', index=False)
test_df.to_csv('../data/emrqa_test.csv', index=False)
