import subprocess


# Run the BERTs

# Ideal-BERT
# Context-BERT
# TF-IDF-BERT

subprocess.run(['nohup', 'python', 'bert_baseline.py'])
subprocess.run(['nohup', 'python', 'bert_context.py'])
subprocess.run(['nohup', 'python', 'experiment_tfidf_bert.py'])
