from utils.vocab import Vocab
import numpy as np
from utils.similarity import evaluate_similarity

def sample_rare_testwords():
    """sample frequent word pairs and rare word pairs from test set"""
    vocab = Vocab(corpus="./data/treebank.txt")
    token_freq = dict(vocab.token_freq)
    with open("data/test.txt") as f:
        lines = f.read().split("\n")
    X = [line.split(" ") for line in lines if len(line) > 0]
    y = list(reversed([i / len(X) for i in range(len(X))]))
    frequent_index = np.array([token_freq[words[0]]+token_freq[words[1]] for words in X])
    lower_bound = np.percentile(frequent_index, 25)
    upper_bound = np.percentile(frequent_index, 75)
    print(lower_bound,upper_bound)
    rare_X = [X[i] for i in range(len(X)) if frequent_index[i]<=lower_bound]
    frequent_X = [X[i] for i in range(len(X)) if frequent_index[i]>=upper_bound]
    rare_y = [y[i] for i in range(len(X)) if frequent_index[i]<=lower_bound]
    frequent_y = [y[i] for i in range(len(X)) if frequent_index[i]>=upper_bound]
    return frequent_X, frequent_y, rare_X, rare_y

def evaluate_similarity2(model, freq_X, freq_y, rare_X, rare_y):
    """Respectively evaluate the frequent and rare word pairs' similarity correlation"""
    spearman_result1, pearson_result1 = evaluate_similarity(model, freq_X, freq_y)
    print(f"spearman correlation: {spearman_result1.correlation:.3f}")
    print(f"pearson correlation: {pearson_result1[0]:.3f}")
    spearman_result2, pearson_result2 = evaluate_similarity(model, rare_X, rare_y)
    print(f"spearman correlation: {spearman_result2.correlation:.3f}")
    print(f"pearson correlation: {pearson_result2[0]:.3f}")


