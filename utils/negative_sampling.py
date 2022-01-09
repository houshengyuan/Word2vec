import random
import numpy as np
from  math import pow,sqrt
from utils.vocab import Vocab

class NEG:
    def __init__(self, vocab: Vocab, alpha: float=0.75, size: int=20, subsampling: bool=False, subsample_thr:float = 1e-3):
        random.seed(42)
        np.random.seed(42)
        self.alpha = alpha
        self.vocab = vocab
        self.size = size
        self.subsample_thr = subsample_thr
        self.table_size = int(1e6)
        self.subsampling = subsampling
        self.sample_table = self.build_sampler(self.vocab,self.alpha,self.subsampling)

    def build_sampler(self,vocab: Vocab,alpha: float,subsampling: bool=False):
        if subsampling:
            freq = np.array(list(map(lambda x: x[1], vocab.token_freq)))
            selection = list(map(lambda x: sqrt(self.subsample_thr / x) + self.subsample_thr / x, freq))
            sifter = []
            for i, t in enumerate(selection):
                shaizi = random.random()
                if shaizi > t:
                    sifter.append(0)
                else:
                    sifter.append(1)
            sifter = np.array(sifter)
            freq = freq * sifter
            sampler = np.array(list(map(lambda x: pow(x, alpha), freq)))
        else:
            sampler = np.array(list(map(lambda x: pow(x[1], alpha), vocab.token_freq)))
        dominator = np.sum(sampler)
        sampler[:] /= dominator
        sampler[:] = np.cumsum(sampler)
        sampler[-1] = 1.0
        table = []
        st = 0; step = 1.0/self.table_size
        table.append(st)
        for i in range(1,self.table_size+1):
            while i*step>sampler[st]:
                st+=1
            table.append(st)
        return table

    def sample(self,target_id:int):
        sample = []
        for i in range(self.size):
            random_number = random.randint(0,self.table_size-1)
            id = self.sample_table[random_number]
            if id != target_id:
                sample.append(id)
        if len(sample)==0:
           sample.append((target_id+1)%len(self.vocab))
        return np.array(sample)

