import random
import sys
import time

import numpy as np
import argparse

from utils.vocab import Vocab
from word2vec import CBOW,Skipgram

# Check Python Version
assert sys.version_info[0] == 3
assert sys.version_info[1] >= 6


def test1(args):
    random.seed(42)
    np.random.seed(42)

    vocab = Vocab(corpus="./data/debug.txt")
    if args.task.lower() == "cbow":
        model=  CBOW(vocab, vector_dim=4, negative_sampling=args.neg, hierarchical_softmax=args.hierarchical, size=args.sample_size,
                     subsampling = args.sub_sampling, subsample_thr=args.subsample_thr)
    elif args.task.lower() == "skip-gram":
        model = Skipgram(vocab, vector_dim=4, negative_sampling=args.neg, hierarchical_softmax=args.hierarchical, size=args.sample_size,
                         subsampling = args.sub_sampling, subsample_thr=args.subsample_thr)
    else:
        model = None
    model.train(corpus="./data/debug.txt", window_size=3, train_epoch=10, learning_rate=1.0)

    print(model.most_similar_tokens("i", 5))
    print(model.most_similar_tokens("he", 5))
    print(model.most_similar_tokens("she", 5))

    # 注：如果实现正确，那么最终的loss将会停留在1.0左右，且'i','he','she'三者的相似性较高。


def test2(args):
    random.seed(args.seed)
    np.random.seed(args.seed)

    try:
        if args.task.lower() == "cbow":
            model = CBOW.load_model("tmp/ckpt",hierarchical_softmax=args.hierarchical,negative_sampling=args.neg,size=args.sample_size,
                                    subsampling = args.sub_sampling,subsample_thr=args.subsample_thr)
        elif args.task.lower() == "skip-gram":
            model = Skipgram.load_model("tmp/ckpt",hierarchical_softmax=args.hierarchical,negative_sampling=args.neg,size=args.sample_size,
                                    subsampling = args.sub_sampling,subsample_thr=args.subsample_thr)
        else:
            model = None
    except FileNotFoundError:
        vocab = Vocab(corpus="./data/treebank.txt", max_vocab_size=-1)
        if args.task.lower() == "cbow":
            model = CBOW(vocab, vector_dim=args.vector_dim, hierarchical_softmax=args.hierarchical,negative_sampling=args.neg, size=args.sample_size,
                         subsampling = args.sub_sampling, subsample_thr=args.subsample_thr)
        elif args.task.lower() == "skip-gram":
            model = Skipgram(vocab, vector_dim=args.vector_dim,hierarchical_softmax=args.hierarchical,negative_sampling=args.neg, size=args.sample_size,
                             subsampling = args.sub_sampling, subsample_thr=args.subsample_thr)
        else:
            model = None

    start_time = time.time()
    model.train(corpus="./data/treebank.txt", window_size=args.window_size, train_epoch=args.epoch, learning_rate=args.learning_rate, save_path="tmp/ckpt")
    end_time = time.time()

    print(f"Cost {(end_time - start_time) / 60:.1f} min")
    print(model.most_similar_tokens("i", 10))



def test3(args):
    from utils.similarity import get_test_data, evaluate_similarity

    model =  None
    if  args.task.lower() == "cbow":
        model = CBOW.load_model("tmp/ckpt",hierarchical_softmax=args.hierarchical,negative_sampling=args.neg,size=args.sample_size, subsampling = args.sub_sampling)
    elif args.task.lower() == "skip-gram":
        model = Skipgram.load_model("tmp/ckpt",hierarchical_softmax=args.hierarchical,negative_sampling=args.neg,size=args.sample_size, subsampling = args.sub_sampling)
    spearman_result, pearson_result = evaluate_similarity(model, *get_test_data())
    print(f"spearman correlation: {spearman_result.correlation:.3f}")
    print(f"pearson correlation: {pearson_result[0]:.3f}")


def rarity_test(args):
    from utils.test_subset import sample_rare_testwords,evaluate_similarity2

    model = None
    if args.task.lower() == "cbow":
        model = CBOW.load_model("tmp/ckpt", hierarchical_softmax=args.hierarchical, negative_sampling=args.neg,
                                size=args.sample_size, subsampling=args.sub_sampling)
    elif args.task.lower() == "skip-gram":
        model = Skipgram.load_model("tmp/ckpt", hierarchical_softmax=args.hierarchical, negative_sampling=args.neg,
                                    size=args.sample_size, subsampling=args.sub_sampling)
    evaluate_similarity2(model, *sample_rare_testwords())


def parse_options():
    parser = argparse.ArgumentParser(description='test1')
    parser.add_argument("--seed", type=int, default=42, help="seed for numpy and random")
    parser.add_argument("--vector-dim", type=int, default=12, help="embedding dim for hidden space")
    parser.add_argument("--window-size", type=int, default=4, help="context window size")
    parser.add_argument("--epoch", type=int, default=10, help="epoch")
    parser.add_argument("--learning-rate", type=float, default=1e-2, help="learning rate")
    parser.add_argument("--task", type=str, default="cbow", help="model name")
    parser.add_argument("--hierarchical", action="store_true", help="option for hierarchical softmax")
    parser.add_argument("--neg", action="store_true", help="option for negative sampling")
    parser.add_argument("--sample-size", type=int, default=10, help="size for negative sampling")
    parser.add_argument("--sub-sampling", action="store_true", help="option for subsampling of frequent words")
    parser.add_argument("--subsample-thr", type=float, default=1e-3, help="threshold for subsampling of frequent words")
    args = parser.parse_args()
    return args

def main():
    args = parse_options()
    test1(args)
    test2(args)
    test3(args)
    rarity_test(args)


if __name__ == '__main__':
    main()
