import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer
import warnings

def add_tokenized_column(tokenizer, column):
    map_dict = {}
    for i in df[column].unique():
        map_dict[i] = tokenizer.tokenize(i)
    df[column+"_tokenized"] = df[column].map(map_dict)
    df[column+"_tokenized_len"] = df[column+"_tokenized"].map(len)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    sns.set_theme(style="whitegrid")

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    df = pd.read_csv("../data/qa.csv")
    df = df.dropna() # need to drop na values even though i dropped before saving, interesting

    columns_to_tokenize = ["question", "context", "evidence"]
    for column in columns_to_tokenize:
        add_tokenized_column(tokenizer, column)
    df_stats = pd.concat([df[column + "_tokenized_len"].describe() for column in columns_to_tokenize], axis=1)

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    for i, column in enumerate(columns_to_tokenize):
        sns.distplot(df[column + "_tokenized_len"], ax=axs[i])
        axs[i].set_title(column)
    plt.show()