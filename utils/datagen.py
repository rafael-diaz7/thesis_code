import json
import pandas as pd

with open('../data/data.json') as f:
    data = json.load(f)

question_all = []
context_all = []
answer_all = []
evidence_all = []
answer_start_all = []

for dataset in data['data']:
    if dataset['title'] in ['obesity', 'smoking']:
        continue
    for paragraph in dataset['paragraphs']:
        context = "".join(paragraph['context'])
        for qa_pair in paragraph['qas']:
            questions = list(set(qa_pair['question']))
            for answer in qa_pair['answers']:
                if answer['text'] == "":
                    continue
                answer_start = answer['answer_start']
                answer_start_formatted = answer_start[1] if type(answer_start[1]) != list else answer_start[1][0]
                question_all.extend(questions)
                context_all.extend([context] * len(questions))
                answer_all.extend([answer['text']] * len(questions))
                evidence_all.extend([answer['evidence']] * len(questions))
                answer_start_all.extend([answer_start_formatted] * len(questions))

df = pd.DataFrame({'question': question_all, 'context': context_all, 'answer': answer_all, 'answer_start': answer_start_all, 'evidence': evidence_all})
df = df.dropna()
df.to_csv('../data/emrqa_all_data.csv', index=False)