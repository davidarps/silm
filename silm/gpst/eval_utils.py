import numpy as np
import json
from nltk import Tree
from nltk.metrics import precision, recall, f_measure

class PyNode:
    def __init__(self, left, right, i, j, cache_id) -> None:
        self._i = i
        self._j = j
        self._cache_id = cache_id
        self._decode_cache_id = -1
        self._left = left
        self._right = right
        self._height = 0
        if left is not None and right is not None:
            self._height = max(left._height, right._height) + 1
        self.label = None
        if left is not None and right is not None:
            self._seq_len = left.seq_len + right.seq_len
        else:
            self._seq_len = 1
            
    @property
    def height(self):
        return self._height
        
    @property
    def seq_len(self):
        return self._seq_len

    @property
    def i(self):
        return self._i

    @property
    def j(self):
        return self._j

    @i.setter
    def i(self, v):
        self._i = v

    @j.setter
    def j(self, v):
        self._j = v

    @property
    def pos(self):
        return self._i

    @property
    def is_leaf(self):
        return self._left is None and self._right is None

    @property
    def cache_id(self):
        return self._cache_id

    @property
    def decode_cache_id(self):
        return self._decode_cache_id

    @decode_cache_id.setter
    def decode_cache_id(self, val):
        self._decode_cache_id = val

    @property
    def left(self):
        return self._left

    @property
    def right(self):
        return self._right
    
    def __str__(self):
        if self.left is not None and self.right is not None:
            return f'[{self._cache_id}] ({self.left}, {self.right})'
        else:
            return f'[{self._cache_id}] {self.pos}'

def get_token_tree(root, tokens):
    if root.left is not None and root.right is not None:
        return '({} {})'.format(get_token_tree(root.left, tokens), get_token_tree(root.right, tokens))
    else:
        return tokens[root.pos]

def get_tree_from_merge_trajectory(merge_trajectory: np.array, seq_len, tokens=None, keep_merge_order=False):
    if seq_len == 1:
        return PyNode(None, None, 0, 0, -1)
    spans_for_splits = [[PyNode(None, None, i, i, -1), PyNode(None, None, i + 1, i + 1, -1)]
                        for i in range(seq_len - 1)]
    latest_span = spans_for_splits[0][0] if seq_len > 1 else None
    if keep_merge_order:
        merge_order = []
    for action_i in range(seq_len - 1):
        merge_pos = merge_trajectory[action_i]
        left, right = spans_for_splits[merge_pos]
        if left.i - left.j == 0:
            left.label = tokens[left.i]
        if right.i - right.j == 0:
            right.label = tokens[right.i]
        latest_span = PyNode(left, right, left.i, right.j, -1)
        if left.i - 1 >= 0:
            spans_for_splits[left.i - 1][1] = latest_span
        if right.j < len(spans_for_splits):
            spans_for_splits[right.j][0] = latest_span
        if keep_merge_order:
            merge_order.append(latest_span)
    if keep_merge_order:
        assert len(merge_order) == seq_len - 1
    root = latest_span
    results = [root]
    if tokens is not None:
        if root is not None:
            results.append(get_token_tree(root, tokens))
        else:
            results.append(' '.join(tokens))
    if keep_merge_order:
        results.append(merge_order)
    if len(results) == 1:
        return results[0]
    else:
        return results

def convert_pynode_tree_to_nltk_tree(pynode_tree):
    if pynode_tree.left is None and pynode_tree.right is None:
        return pynode_tree.label
    if pynode_tree.left:
        left = convert_pynode_tree_to_nltk_tree(pynode_tree.left)
    if pynode_tree.right:
        right = convert_pynode_tree_to_nltk_tree(pynode_tree.right)
    label = pynode_tree.label if pynode_tree.label else "S"
    return Tree(label, [left,right])

def load_gold_data(filename, bos_tok="", eos_tok=""):
    
    if "json" in filename:
        with open(filename, "r") as f:
            lines = [json.loads(l) for l in f.read().splitlines()]
        text = [l["text"] for l in lines]
        if "dyck" in filename:
            treestrings = [f'(. {bos_tok} {l["tree"]} {eos_tok} )' for l in lines]
        else:
            treestrings = [l["tree"] for l in lines]
        trees = [Tree.fromstring(s) for s in treestrings]
        return text, trees
    else:
        with open(filename, "r") as f:
            lines = f.read().splitlines()
        return lines, []

# Function to get spans and labels
def get_spans(tree, remove_one_token_spans=True):
    spans = []
    pos = 0  # Position counter

    def traverse(t, start):
        if isinstance(t, str):  # Leaf node
            end = start + 1
            return end
        else:
            cur_start = start
            for child in t:
                start = traverse(child, start)
            end = start
            spans.append((cur_start, end))
            return end

    traverse(tree, 0)
    if remove_one_token_spans:
        spans = set(s for s in spans if (s[1]-s[0])>1)
    else:
        spans = set(spans)
    return spans

def evaluate_trees(gold_trees, pred_trees, include_sentencewise_scores=False):
    scores = {
        "precision": [], 
        "recall": [], 
        "fscore": []
    }
    for splitpoint_tree, gold_tree in zip(pred_trees, gold_trees):

        assert len(splitpoint_tree.leaves()) == len(gold_tree.leaves()), f"{len(splitpoint_tree.leaves())}, {len(gold_tree.leaves())}"
        splitpoint_tree_spans = get_spans(splitpoint_tree)
        gold_tree_spans = get_spans(gold_tree)

        scores["precision"].append(precision(gold_tree_spans, splitpoint_tree_spans))
        scores["recall"].append(recall(gold_tree_spans, splitpoint_tree_spans))
        scores["fscore"].append(f_measure(gold_tree_spans, splitpoint_tree_spans))
    
    result_dict = {
        "precision_mean": np.array(scores["precision"]).mean(),
        "recall_mean": np.array(scores["recall"]).mean(),
        "fscore_mean": np.array(scores["fscore"]).mean(),
    }
    if include_sentencewise_scores:
        result_dict["sentencelevel_precision"]: scores["precision"]
        result_dict["sentencelevel_recall"]: scores["recall"]
        result_dict["sentencelevel_fscore"]: scores["fscore"]
    
    return result_dict