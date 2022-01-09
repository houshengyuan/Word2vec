import matplotlib.pyplot as plt
import numpy as np
from utils.vocab import tokenizer,Vocab

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus']=False

def doc_length(path):
    with open(path,encoding="utf8") as f:
        len_set = []
        for line in f:
            tokens = tokenizer(line)
            len_set.append(len(tokens))
        len_set = np.array(len_set)
        print(np.max(len_set))
        print(np.min(len_set))
        print(np.average(len_set))
        print(len_set.shape)
        plt.hist(len_set,bins=53,density=0,facecolor='orange',edgecolor='black',alpha=0.7,stacked=True)
        plt.title("Distribution of sentence length")
        plt.xlabel("Length")
        plt.ylabel("Number")
        plt.savefig("senc_length.jpg",dpi=600)
        plt.show()

def sentlen(path):
    vocab = Vocab(corpus=path, max_vocab_size=-1)
    vocab_len = [x[1] for x in vocab.token_freq]
    plt.hist(vocab_len, bins=50, range=(0,200),density=0, facecolor='blue', edgecolor='black', alpha=0.7, stacked=True,)
    plt.title("Distribution of word frequency")
    plt.xlabel("Counts")
    plt.ylabel("Frequency")
    plt.savefig("senc_length.jpg", dpi=600)
    plt.show()

def line_plot():
    x = [1,2,4,6,8]
    y1_skip_gram_s = [0.284,0.288,0.280,0.270,0.264]
    y1_skip_gram_p = [0.326, 0.335, 0.328, 0.313, 0.306]
    y2_cbow_s = [0.277,0.319,0.366,0.376,0.383]
    y2_cbow_p = [0.325, 0.375, 0.439, 0.449, 0.451]
    plt.figure(figsize=(7,6),dpi=600)
    plt.xticks(fontsize=20)
    plt.ylim((0.20,0.40))
    plt.yticks([0.20,0.25,0.30,0.35,0.40],fontsize=20)
    plt.xlabel('Window Size',fontsize=17)
    plt.ylabel('Correlation',fontsize=17)
    plt.plot(x, y1_skip_gram_s,color='red', linestyle='-', linewidth=2, marker='^',markerfacecolor='none', markersize=6 ,label='Spearman')
    plt.plot(x, y1_skip_gram_p, color='#0a5f38', linestyle='-', linewidth=2, marker='s', markerfacecolor='none', markersize=6, label='Pearson')
    plt.legend(loc='upper right',fontsize=20)
    plt.show()

def bar_chart():
    x = np.array(list(range(4)))
    y1 = [20.3, 8.6, 10.1, 9.0]
    y2 = [21, 9.0, 7.7, 9.2]
    width = 0.4
    x = x - 0.2
    plt.figure(figsize=(8, 6))
    plt.bar(x, y1, width=width, label='CBOW', edgecolor='black', alpha=0.7, hatch="/")
    plt.bar(x + width, y2, width=width, label='Skip-gram', edgecolor='black', hatch='x')
    plt.xticks(x + 0.2, ['Naive', 'NEG', 'Hierarchical', 'NEG-Subsample'],fontsize=14)
    plt.yticks(fontsize=20)
    plt.xlabel("Model",fontsize=17)
    plt.ylabel("Running time/epoch", fontsize=17)
    plt.legend(loc=1,fontsize=13)  # 设置图例位置plt.ylabel('我是Y轴',fontsize=15)plt.xlabel('我是X轴',fontsize=15)plt.title("基础柱状图——分组柱状图——填充",fontsize=17)plt.show()
    plt.show()

if __name__=="__main__":
    sentlen(path = "data/treebank.txt")



