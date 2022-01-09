import os
import pickle
import time
from os.path import join
from typing import List

import numpy as np

from utils.dataset import Dataset
from utils.vocab import Vocab
from utils.hierarchical_softmax import Huffman_tree
from utils.negative_sampling import NEG

def one_hot(dim: int, idx: int):
    """ Get one-hot vector """
    v = np.zeros(dim)
    v[idx] = 1
    return v


def softmax(x, dim: [int]=-1):
    e_x = np.exp(x - np.max(x, dim))
    return e_x / np.sum(e_x, dim)

def sigmoid(x):
    mask = (x>0)
    pos = (x+np.fabs(x))/2
    neg = (x-np.fabs(x))/2
    return mask*(1/(1+np.exp(-pos)))+(1-mask)*(np.exp(neg)/(1+np.exp(neg)))

class CBOW:
    def __init__(self, vocab: Vocab, vector_dim: int,
                 hierarchical_softmax: bool=False, negative_sampling: bool=False,
                 size: int=10, subsampling: bool=False, subsample_thr: float=1e-3):
        self.vocab = vocab
        self.vector_dim = vector_dim
        self.hierarchical_softmax = hierarchical_softmax
        self.negative_sampling = negative_sampling
        self.subsampling = subsampling
        self.subsample_thr = subsample_thr
        os.makedirs("log", exist_ok=True)
        self.log = open("log/cbow" + ("_hierarchical" if self.hierarchical_softmax else "") + ("_neg" if self.negative_sampling else "")
                        + ("_sub" if self.subsampling else "") + ".txt", "a+")

        self.W1 = np.random.uniform(-1, 1, (len(self.vocab), self.vector_dim))  # V x N
        if self.hierarchical_softmax:
            self.tree = Huffman_tree(self.vocab, dim=self.vector_dim)
        else:
            self.W2 = np.random.uniform(-1, 1, (len(self.vocab), self.vector_dim))  # N x V
        if negative_sampling:
            self.sampler=NEG(vocab=self.vocab, alpha=0.75, size=size, subsampling = self.subsampling, subsample_thr = self.subsample_thr)

    def train(self, corpus: str, window_size: int, train_epoch: int, learning_rate: float, save_path: str = None):
        dataset = Dataset(corpus, window_size, "CBOW")

        for epoch in range(1, train_epoch + 1):
            start_time = time.time()
            avg_loss = self.train_one_epoch(dataset, learning_rate)
            end_time = time.time()
            print(f"Epoch {epoch}, loss: {avg_loss:.2f}. Cost {(end_time - start_time) / 60:.1f} min",file=self.log,flush=True)
            if save_path is not None:
                self.save_model(save_path)

    def train_one_epoch(self, dataset: Dataset, learning_rate: float):
        steps, total_loss = 0, 0.0

        for steps, sample in enumerate(iter(dataset), start=1):
            context_tokens, target_token = sample
            loss = self.train_one_step(steps,context_tokens, target_token, learning_rate)
            total_loss += loss

            if steps % 10000 == 0:
                print(f"Step: {steps}. Avg. loss: {total_loss / steps: .2f}",file=self.log,flush=True)

        return total_loss / steps

    def train_one_step(self, step:int ,context_tokens: List[str], target_token: str, learning_rate: float) -> float:
        """
        Predict the probability of the target token given context tokens.

        :param context_tokens:  List of tokens around the target token
        :param target_token:    Target (center) token
        :param learning_rate:   Learning rate of each step
        :param hierarchical_softmax:    whether use the hierarchical softmax technique
        :param negative_sampling:    whether use the negative sampling technique
        :return:    loss of the target token
        """
        if self.hierarchical_softmax:
            """CBOW with hierarchical softmax"""
            # ==== Construct one-hot vectors ====
            context_ids = [self.vocab.token_to_idx(i) for i in context_tokens]
            input_onehot = np.array(list(map(lambda x: one_hot(len(self.vocab), x), context_ids)))
            # ==== Forward step ====
            assert input_onehot.shape[-1] == self.W1.shape[0]
            word_vector = input_onehot @ self.W1
            word_vector = np.average(word_vector, axis=0)
            target_id = self.vocab.token_to_idx(target_token)
            target_path = self.tree.get_nodepath(target_id)
            context_vector = np.array([target_path[i].vector for i in range(len(target_path)-1)])
            context_one_hot = np.array([0 if target_path[i + 1].direction == 0 else 1 for i in range(len(target_path) - 1)])
            logit = context_vector @ word_vector
            output = sigmoid(logit)
            # ==== Calculate loss ====
            loss = -np.sum(np.log((-2 * context_one_hot + 1) * output + context_one_hot + 1e-8))
            # ==== Update parameters ====
            self.W1[context_ids] -= learning_rate * ((output - 1 + context_one_hot) @ context_vector) / len(context_tokens)
            for j, node in enumerate(target_path):
                if target_path[j].vector is not None:
                    target_path[j].vector -= learning_rate * (output[j] - 1 + context_one_hot[j]) * word_vector
        elif self.negative_sampling:
            """CBOW with nagative sampling"""
            # ==== Construct one-hot vectors ====
            context_ids = [self.vocab.token_to_idx(i) for i in context_tokens]
            input_onehot = np.array(list(map(lambda x: one_hot(len(self.vocab), x), context_ids)))
            # ==== Forward step ====
            assert input_onehot.shape[-1] == self.W1.shape[0]
            word_vector = input_onehot @ self.W1
            word_vector = np.average(word_vector, axis=0)
            target_id = self.vocab.token_to_idx(target_token)
            positive_logit = word_vector @ self.W2[target_id]
            negative_ids = self.sampler.sample(target_id)
            negative_logit = -self.W2[negative_ids] @ word_vector
            positive_output = sigmoid(positive_logit)
            negative_output = sigmoid(negative_logit)
            # ==== Calculate loss ====
            loss = -np.log(positive_output+1e-8)-np.sum(np.log(negative_output+1e-8))
            # ==== Update parameters ====
            self.W1[context_ids] -= learning_rate * ((positive_output-1) * self.W2[target_id]+(1-negative_output)@self.W2[negative_ids]) / len(context_ids)
            self.W2[target_id] -= learning_rate * (positive_output-1) * word_vector
            self.W2[negative_ids] -= learning_rate * np.outer(1-negative_output,word_vector)
        else:
            """naive CBOW"""
            # ==== Construct one-hot vectors ====
            context_ids = [self.vocab.token_to_idx(i) for i in context_tokens]
            input_onehot = np.array(list(map(lambda x: one_hot(len(self.vocab), x), context_ids)))
            # ==== Forward step ====
            assert input_onehot.shape[-1] == self.W1.shape[0]
            word_vector = input_onehot @ self.W1
            word_vector = np.average(word_vector, axis=0)
            logit =  self.W2 @word_vector
            output = softmax(logit)
            # ==== Calculate loss ====
            target_id = self.vocab.token_to_idx(target_token)
            loss = -np.log(output[target_id]+1e-8)
            # ==== Update parameters ====
            target_one_hot = one_hot(len(self.vocab), target_id)
            self.W1[context_ids] -= learning_rate * (output @ self.W2 - self.W2[target_id]) / len(context_ids)
            self.W2[:] -= learning_rate * np.outer(output - target_one_hot, word_vector)
        return loss

    def similarity(self, token1: str, token2: str):
        """ Calculate cosine similarity of token1 and token2 """
        v1 = self.W1[self.vocab.token_to_idx(token1)]
        v2 = self.W1[self.vocab.token_to_idx(token2)]
        v1 = v1 / np.linalg.norm(v1)
        v2 = v2 / np.linalg.norm(v2)
        return np.dot(v1, v2)

    def most_similar_tokens(self, token: str, n: int):
        """ Find the n words most similar to the given token """
        norm_W1 = self.W1 / np.linalg.norm(self.W1, axis=1, keepdims=True)

        idx = self.vocab.token_to_idx(token, warn=True)
        v = norm_W1[idx]

        cosine_similarity = np.dot(norm_W1, v)
        nbest_idx = np.argsort(cosine_similarity)[-n:][::-1]

        results = []
        for idx in nbest_idx:
            _token = self.vocab.idx_to_token(idx)
            results.append((_token, cosine_similarity[idx]))

        return results

    def save_model(self, path: str):
        """ Save model and vocabulary to `path` """
        os.makedirs(path, exist_ok=True)
        self.vocab.save_vocab(path)

        with open(join(path, "cbow"+("_hierarchical" if self.hierarchical_softmax else "")
            +("_neg" if self.negative_sampling else "")+ ("_sub" if self.subsampling else "") + ".pkl"), "wb") as f:
            if self.hierarchical_softmax:
                param = {"W1": self.W1, "tree": self.tree}
            else:
                param = {"W1": self.W1, "W2": self.W2}
            pickle.dump(param, f)

        print(f"Save model to {path}",file=self.log,flush=True)

    @classmethod
    def load_model(cls, path: str, hierarchical_softmax:bool= False, negative_sampling:bool= False,
                   size:int=10, subsampling:bool= False, subsample_thr:float= 1e-3):
        """ Load model and vocabulary from `path` """
        vocab = Vocab.load_vocab(path)

        with open(join(path, "cbow" + ("_hierarchical" if hierarchical_softmax else "")
                             + ("_neg" if negative_sampling else "") + ("_sub" if subsampling else "") + ".pkl"), "rb") as f:
            param = pickle.load(f)

        if hierarchical_softmax:
            W1, tree = param["W1"], param["tree"]
            model = cls(vocab, W1.shape[1], hierarchical_softmax=hierarchical_softmax, negative_sampling=negative_sampling, size=size, subsampling=subsampling,subsample_thr=subsample_thr)
            model.W1, model.tree = W1, tree
        else:
            W1, W2 = param["W1"], param["W2"]
            model = cls(vocab, W1.shape[1], hierarchical_softmax=hierarchical_softmax, negative_sampling=negative_sampling, size=size, subsampling=subsampling,subsample_thr=subsample_thr)
            model.W1, model.W2 = W1, W2

        print(f"Load model from {path}")
        return model


class Skipgram:
    def __init__(self, vocab: Vocab, vector_dim: int,hierarchical_softmax: bool=False, negative_sampling: bool=False, size: int=10, subsampling: bool=False, subsample_thr: float=1e-3):
        self.vocab = vocab
        self.vector_dim = vector_dim
        self.hierarchical_softmax = hierarchical_softmax
        self.negative_sampling = negative_sampling
        self.subsampling = subsampling
        self.subsample_thr = subsample_thr
        os.makedirs("log", exist_ok=True)
        self.log = open("log/skip-gram"+("_hierarchical" if self.hierarchical_softmax else "") + ("_neg" if self.negative_sampling else "")
                        + ("_sub" if self.subsampling else "") + ".txt","a+")

        self.W1 = np.random.uniform(-1, 1, (len(self.vocab), self.vector_dim))  # V x N
        if self.hierarchical_softmax:
            self.tree = Huffman_tree(self.vocab, dim=vector_dim)
        else:
            self.W2 = np.random.uniform(-1, 1, (len(self.vocab), self.vector_dim))  # N x V
        if negative_sampling:
            self.sampler = NEG(vocab=self.vocab, alpha=0.75, size=size, subsampling=self.subsampling, subsample_thr=self.subsample_thr)

    def train(self, corpus: str, window_size: int, train_epoch: int, learning_rate: float, save_path: str = None):
        dataset = Dataset(corpus, window_size, "skip-gram")

        for epoch in range(1, train_epoch + 1):
            start_time = time.time()
            avg_loss = self.train_one_epoch(dataset, learning_rate)
            end_time = time.time()
            print(f"Epoch {epoch}, loss: {avg_loss:.2f}. Cost {(end_time - start_time) / 60:.1f} min",file=self.log,flush=True)
            if save_path is not None:
                self.save_model(save_path)

    def train_one_epoch(self, dataset: Dataset, learning_rate: float):
        steps, total_loss = 0, 0.0

        for steps, sample in enumerate(iter(dataset), start=1):
            context_tokens, target_token = sample
            loss = self.train_one_step(context_tokens, target_token, learning_rate)
            total_loss += loss

            if steps % 10000 == 0:
                print(f"Step: {steps}. Avg. loss: {total_loss / steps: .2f}",file=self.log,flush=True)

        return total_loss / steps

    def train_one_step(self, context_token: str, target_tokens: List[str], learning_rate: float) -> float:
        """
        Predict the probability of the target token given context tokens.

        :param context_token: Context  (center) token
        :param target_tokens:  List of target tokens around the context token
        :param learning_rate:   Learning rate of each step
        :return:    loss of the target token
        """
        if self.hierarchical_softmax:
            """skip-gram with hierarchical softmax"""
            # ==== Construct one-hot vectors ====
            context_id = self.vocab.token_to_idx(context_token)
            input_onehot = one_hot(len(self.vocab),context_id)
            # ==== Forward step ====
            assert input_onehot.shape[0] == self.W1.shape[0]
            word_vector = input_onehot @ self.W1
            target_ids = [self.vocab.token_to_idx(i) for i in target_tokens]
            target_path = [self.tree.get_nodepath(i) for i in target_ids]
            loss = 0.0
            target_gradient = []
            for i,path in enumerate(target_path):
                gradient = []
                context_vector = np.array([path[i].vector for i in range(len(path)-1)])
                context_one_hot = np.array([0 if path[i+1].direction==0 else 1 for i in range(len(path)-1)])
                logit = context_vector @ word_vector
                output = sigmoid(logit)
                # ==== Calculate loss ====
                loss += -np.sum(np.log((-2*context_one_hot + 1) * output + context_one_hot + 1e-8))/len(target_tokens)
                # ==== Update parameters ====
                self.W1[target_ids[i]] -= learning_rate * ((output - 1 + context_one_hot) @ context_vector)/len(target_tokens)
                for j, node in enumerate(path):
                    if path[j].vector is not None:
                        gradient.append((output[j] - 1 + context_one_hot[j]) * word_vector/len(target_tokens))
                target_gradient.append(gradient)
            for i, path in enumerate(target_path):
                for j, node in enumerate(path):
                    if path[j].vector is not None:
                        path[j].vector -= learning_rate * target_gradient[i][j]
        elif self.negative_sampling:
            # ==== Construct one-hot vectors ====
            context_id = self.vocab.token_to_idx(context_token)
            input_onehot = one_hot(len(self.vocab), context_id)
            # ==== Forward step ====
            assert input_onehot.shape[0] == self.W1.shape[0]
            word_vector = input_onehot @ self.W1
            target_ids = [self.vocab.token_to_idx(i) for i in target_tokens]
            loss = 0.0
            for i in target_ids:
                negative_ids = self.sampler.sample(i)
                positive_logit = word_vector @ self.W2[i]
                negative_logit = -self.W2[negative_ids] @ word_vector
                positive_output = sigmoid(positive_logit)
                negative_output = sigmoid(negative_logit)
                # ==== Calculate loss ====
                loss += (-np.log(positive_output+1e-8)-np.sum(np.log(negative_output+1e-8)))/len(target_ids)
                # ==== Update parameters ====
                self.W1[context_id] -= learning_rate*(self.W2[i]*(positive_output-1)+(1-negative_output)@self.W2[negative_ids])/len(target_tokens)
                self.W2[i] -= learning_rate*word_vector*(positive_output-1)/len(target_tokens)
                self.W2[negative_ids] -= learning_rate*np.outer(1-negative_output,word_vector)/len(target_tokens)
        else:
            # ==== Construct one-hot vectors ====
            context_id = self.vocab.token_to_idx(context_token)
            input_onehot = one_hot(len(self.vocab), context_id)
            # ==== Forward step ====
            assert input_onehot.shape[0] == self.W1.shape[0]
            word_vector = input_onehot @ self.W1
            logit = self.W2 @ word_vector
            output = softmax(logit)
            target_ids = [self.vocab.token_to_idx(i) for i in target_tokens]
            # ==== Calculate loss ====
            loss = np.average(-np.log(output[target_ids]+1e-8))
            # ==== Update parameters ====
            W1_grad = np.zeros(self.W1.shape)
            W1_grad[context_id] += -np.average(self.W2[target_ids], axis=0) + output @ self.W2
            W2_grad = np.outer(output, word_vector)
            W2_grad[target_ids] -= np.expand_dims(word_vector, axis=0) / len(target_ids)
            assert list(W1_grad.shape) == list(self.W1.shape)
            assert list(W2_grad.shape) == list(self.W2.shape)
            self.W1[:] -= learning_rate * W1_grad
            self.W2[:] -= learning_rate * W2_grad
        return loss

    def similarity(self, token1: str, token2: str):
        """ Calculate cosine similarity of token1 and token2 """
        v1 = self.W1[self.vocab.token_to_idx(token1)]
        v2 = self.W1[self.vocab.token_to_idx(token2)]
        v1 = v1 / np.linalg.norm(v1)
        v2 = v2 / np.linalg.norm(v2)
        return np.dot(v1, v2)

    def most_similar_tokens(self, token: str, n: int):
        """ Find the n words most similar to the given token """
        norm_W1 = self.W1 / np.linalg.norm(self.W1, axis=1, keepdims=True)

        idx = self.vocab.token_to_idx(token, warn=True)
        v = norm_W1[idx]

        cosine_similarity = np.dot(norm_W1, v)
        nbest_idx = np.argsort(cosine_similarity)[-n:][::-1]

        results = []
        for idx in nbest_idx:
            _token = self.vocab.idx_to_token(idx)
            results.append((_token, cosine_similarity[idx]))

        return results

    def save_model(self, path: str):
        """ Save model and vocabulary to `path` """
        os.makedirs(path, exist_ok=True)
        self.vocab.save_vocab(path)

        with open(join(path, "skip-gram" + ("_hierarchical" if self.hierarchical_softmax else "")
                             + ("_neg" if self.negative_sampling else "") + ("_sub" if self.subsampling else "") + ".pkl"), "wb") as f:
            if self.hierarchical_softmax:
                param = {"W1": self.W1, "tree": self.tree}
            else:
                param = {"W1": self.W1, "W2": self.W2}
            pickle.dump(param, f)

        print(f"Save model to {path}",file=self.log,flush=True)

    @classmethod
    def load_model(cls, path: str, hierarchical_softmax:bool= False, negative_sampling:bool= False,
                   size:int= 10, subsampling:bool= False, subsample_thr:float= 1e-3):
        """ Load model and vocabulary from `path` """
        vocab = Vocab.load_vocab(path)

        with open(join(path, "skip-gram" + ("_hierarchical" if hierarchical_softmax else "")
                             + ("_neg" if negative_sampling else "") + ("_sub" if subsampling else "") + ".pkl"), "rb") as f:
            param = pickle.load(f)

        if hierarchical_softmax:
            W1, tree = param["W1"], param["tree"]
            model = cls(vocab, W1.shape[1], hierarchical_softmax=hierarchical_softmax, negative_sampling=negative_sampling, size=size, subsampling=subsampling, subsample_thr=subsample_thr)
            model.W1, model.tree = W1, tree
        else:
            W1, W2 = param["W1"], param["W2"]
            model = cls(vocab, W1.shape[1], hierarchical_softmax=hierarchical_softmax, negative_sampling=negative_sampling, size=size, subsampling=subsampling, subsample_thr=subsample_thr)
            model.W1, model.W2 = W1, W2

        print(f"Load model from {path}")
        return model