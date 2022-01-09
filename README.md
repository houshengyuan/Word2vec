# Word2vec
  
## Description  
  
This project implements eight different versions of word2vec, including:
  
* Original CBOW/Skip-gram
* CBOW/Skip-gram with Hierarchical Softmax
* CBOW/Skip-gram with Negative Sampling
* CBOW/Skip-gram with Negative Sampling+Subsampling of Frequent words

## Environmental Requirements:
  
* Python Version >= 3.6
* Numpy   
* Scipy  
  
## Task:
  
* test1: Debug on dataset 'data/debug.txt'. 
* test2: Train model on dataset 'data/treebank.txt'. 
* test3: Test on whole testset w.r.t Pearson and Spearman Correlation.
* test4: Test performance on frequent/rare word pairs inn testing set.
  
## Options
  

## Test Examples
  
Original CBOW/Skip-gram
```
python main.py --task cbow
```
```
python main.py --task skip-gram
```
CBOW/Skip-gram with Hierarchical Softmax
```
python main.py --task cbow --hierarchical
```
```
python main.py --task skip-gram --hierarchical
```
CBOW/Skip-gram with Negative Sampling
```
python main.py --task cbow --neg --sample-size 5
```
```
python main.py --task skip-gram --neg --sample-size 5
```
CBOW/Skip-gram with Negative Sampling+Subsampling of Frequent words
```
python main.py --task cbow --neg --sample-size 5 --sub-sampling --subsample-thr 1e-3
```
```
python main.py --task skip-gram --neg --sample-size 5 --sub-sampling --subsample-thr 1e-3
```
