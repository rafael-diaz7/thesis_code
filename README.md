# Methods for Large Context Clinical Question Answering

This was my thesis presented to Christopher Newport University in 2024 for my Masters of Science in Computer Science.

## Abstract
Natural Language Processing (NLP) enables interaction between humans and software using language. Extractive Question Answering (QA) within NLP provides answers to user queries based on a given context, and it is particularly useful in clinical domains where Electronic Medical Records (EMRs) are complex. Common QA techniques require adaptation to handle the medical jargon and extensive length of EMRs. This research compares six approaches, using either BERT or Llama2, to identify the most effective method for extractive QA in large contexts. Two approaches serve as baselines. The highest-performing method utilized BERT to first identify sentences as evidence, which were then processed by another BERT model to extract the answer, achieving an F1 score of 34.2 with a recall of 40.57. While Llama 2 exhibited higher recall, it was quite verbose. The study suggests that specialized models may outperform larger, general models, and a combination could optimize performance.

## More info
Please feel free to reach out to me to view the full thesis, I can provide both the LaTeX used to generate my paper as well as a .pdf format of it.
It is also available on ProQuest at this link here: https://www.proquest.com/docview/3198981407?sourcetype=Dissertations%20&%20Theses
