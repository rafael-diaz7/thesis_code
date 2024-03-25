import pandas as pd
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

train_df = pd.read_csv('../data/emrqa_train.csv')
val_df = pd.read_csv('../data/emrqa_val.csv')
test_df = pd.read_csv('../data/emrqa_test.csv')

def adjusted_calc_token_spans(question, evidence, answer):
    inputs = tokenizer(question, evidence, truncation='only_second',
                       stride=50, return_overflowing_tokens=True, return_offsets_mapping=True)
    offset = inputs['offset_mapping'][0]
    calc_answer_start = evidence.find(answer)
    end_char = calc_answer_start + len(answer)
    sequence_ids = inputs.sequence_ids()

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
    return start_position, end_position


def true_calc_token_spans(question, evidence, answer):
    inputs = tokenizer(question, evidence, return_offsets_mapping=True)
    offset = inputs['offset_mapping']
    calc_answer_start = evidence.find(answer)
    end_char = calc_answer_start + len(answer)
    sequence_ids = inputs.sequence_ids()

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

    if start_position == 511 and end_position == 511:
        return 0, 0
    return start_position, end_position

train_df[['start_token', 'end_token']] = train_df.apply(lambda x: adjusted_calc_token_spans(x['question'], x['evidence'], x['answer']), axis=1, result_type='expand')
val_df[['start_token', 'end_token']] = val_df.apply(lambda x: adjusted_calc_token_spans(x['question'], x['evidence'], x['answer']), axis=1, result_type='expand')
test_df[['start_token', 'end_token']] = test_df.apply(lambda x: true_calc_token_spans(x['question'], x['evidence'], x['answer']), axis=1, result_type='expand')

train_df.to_csv('../data/emrqa_evidence_train.csv', index=False)
val_df.to_csv('../data/emrqa_evidence_val.csv', index=False)
test_df.to_csv('../data/emrqa_evidence_test.csv', index=False)

train_df[['start_token', 'end_token']] = train_df.apply(lambda x: adjusted_calc_token_spans(x['question'], x['context'], x['answer']), axis=1, result_type='expand')
val_df[['start_token', 'end_token']] = val_df.apply(lambda x: adjusted_calc_token_spans(x['question'], x['context'], x['answer']), axis=1, result_type='expand')
test_df[['start_token', 'end_token']] = test_df.apply(lambda x: true_calc_token_spans(x['question'], x['context'], x['answer']), axis=1, result_type='expand')

train_df.to_csv('../data/emrqa_context_train.csv', index=False)
val_df.to_csv('../data/emrqa_context_val.csv', index=False)
test_df.to_csv('../data/emrqa_context_test.csv', index=False)