import json
from abc import ABC
import datasets


class EmrqaDataset(datasets.GeneratorBasedBuilder, ABC):

    VERSION = datasets.Version("1.0.0")
    BUILDER_CONFIGS = [
        datasets.BuilderConfig(name="emrqa_context", version=VERSION, description="EMRQA dataset with context and no "
                                                                                  "evidence"),
        datasets.BuilderConfig(name="emrqa_evidence", version=VERSION, description="EMRQA dataset with only evidence")
    ]
    DESCRIPTION = "EMRQA dataset"
    DEFAULT_CONFIG_NAME = "emrqa_context"

    def _info(self):
        return datasets.DatasetInfo(
            description=self.DESCRIPTION,
            features=datasets.Features(
                {
                    "question": datasets.Value("string"),
                    "context": datasets.Value("string"),
                    "answers": datasets.Sequence(
                        {
                            "text": datasets.Value("string"),
                            "answer_start": datasets.Value("int32"),
                            "evidence_start": datasets.Value("int32"),
                            "evidence": datasets.Value("string"),
                        }
                    )
                }
            ),
            supervised_keys=None
        )

    def _generate_examples(self, filepath):
        with open('../data/data.json') as f:
            for dataset in json.load(f)['data']:
                if dataset['title'] in ['obesity', 'smoking']:
                    continue
                for paragraph in dataset['paragraphs']:
                    context = "".join(paragraph['context'])
                    for qa_pair in paragraph['qas']:
                        questions = list(set(qa_pair['question']))
                        answer = qa_pair['answers'][0]
                        if answer['text'] == "":
                            continue
                        answer_start = answer['answer_start']
                        answer_start_formatted = answer_start[1] if type(answer_start[1]) != list else answer_start[1][0]
                        yield {
                            questions[0],
                            context,
                            {
                                answer['text'],
                                answer_start_formatted,
                                answer['evidence'],
                                answer['evidence_start']
                            }
                        }

# train test 80/20 split, 90/10 split for train/val split.
# do one question from qa_pair to lower the number of samples (2 million to 90k)
# don't use SQuAD, just use the emrQA dataset