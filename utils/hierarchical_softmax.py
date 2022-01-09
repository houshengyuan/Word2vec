import heapq
import numpy as np

from utils.vocab import Vocab

class Huffman_node:
    """
    Huffman node
    self.direction: the current node is left(0)/right(1) child of its parent node
    self.node: List of its left and right child
    self.freq: the Huffman weight value
    self.vector: context representation vector of this node
    """
    def __init__(self, leftnode = None, rightnode = None, leaf = None, vector_dim = None):
        self.direction = None
        if leftnode is not None and rightnode is not None:
            assert isinstance(leftnode, Huffman_node)
            assert isinstance(rightnode, Huffman_node)
            leftnode.direction = 0
            rightnode.direction = 1
            self.node = [leftnode, rightnode]
            self.freq = leftnode.get_freq() + rightnode.get_freq()
            self.vector = np.random.uniform(-1, 1, (vector_dim))
        else:
            assert leaf is not None
            self.node = leaf[0]
            self.freq = leaf[1]
            self.vector = None

    def __lt__(self, other):
        return self.freq < other.freq

    def get_node(self):
        return self.node

    def get_freq(self)-> int:
        return self.freq


class Huffman_tree:
    """
    Huffman tree
    self.path: Root Path Lists of every leaf node
    """
    def __init__(self, vocab:Vocab, dim:int =4):
        self.vocab = vocab
        self.root = self.build_tree(self.vocab,dim)
        self.path = self.leafnode_path()

    def build_tree(self, vocab:Vocab, dim:int):
        """Construct the set of tree nodes"""
        node = list(map(lambda x:Huffman_node(leaf=x), vocab.token_freq))
        heapq.heapify(node)
        while len(node)>1:
            node1 = heapq.heappop(node)
            node2 = heapq.heappop(node)
            heapq.heappush(node,Huffman_node(leftnode = node1, rightnode = node2, vector_dim=dim))
        return node[0]

    def get_nodepath(self,target_id:int)->list:
        return self.path[target_id]

    def leafnode_path(self):
        """find the node path of every leaf and store"""
        stack_node = []
        stack_path = []
        path = {}
        stack_node.append(self.root)
        stack_path.append([self.root])
        while len(stack_node)!=0:
            tmp_node = stack_node.pop()
            tmp_path = stack_path.pop()
            if isinstance(tmp_node.node,str):
                path[self.vocab.token_to_idx(tmp_node.node)] = tmp_path
            else:
                stack_node.append(tmp_node.node[1])
                stack_node.append(tmp_node.node[0])
                stack_path.append(tmp_path + [tmp_node.node[1]])
                stack_path.append(tmp_path + [tmp_node.node[0]])
        return path
