from tqdm import tqdm
import codecs
import torch
from torch.utils import data
import os
from silm.gpst.misc import *
import json
from nltk import Tree
    
class TextDataset(data.Dataset):
    
    def __init__(self, file_path, json_col="text"):
        """
        sentence_start: the stripped article must start with a complete sentence
        """
        self._lines = []
        if not(json_col):
            with codecs.open(file_path, mode='r', encoding='utf-8') as f_in:
                for line in f_in:
                    self._lines.append(line)
        else:
            with codecs.open(file_path, mode='r', encoding='utf-8') as f_in:
                for line in f_in:
                    self._lines.append(json.loads(line)[json_col])
    

    def __len__(self):
        return len(self._lines)

    def __getitem__(self, idx):
        return self._lines[idx]

def convert_to_bracket(root, org_tokens, atom_spans, indices_mapping):
    if [root.i, root.j] in atom_spans:
        assert indices_mapping[root.i] == indices_mapping[root.j]
        return f'{org_tokens[indices_mapping[root.i]]}'
    else:
        if root.left is not None and root.right is not None:
            left_str = convert_to_bracket(root.left, org_tokens, atom_spans, indices_mapping)
            right_str = convert_to_bracket(root.right, org_tokens, atom_spans, indices_mapping)
            return f'({left_str} {right_str})'
        else:
            return f'{org_tokens[indices_mapping[root.i]]}'

def convert_to_ptb(root, org_tokens, atom_spans, indices_mapping):
    if [root.i, root.j] in atom_spans:
        assert indices_mapping[root.i] == indices_mapping[root.j]
        return f'(T-1 {org_tokens[indices_mapping[root.i]]})'
    else:
        if root.left is not None and root.right is not None:
            left_str = convert_to_ptb(root.left, org_tokens, atom_spans, indices_mapping)
            right_str = convert_to_ptb(root.right, org_tokens, atom_spans, indices_mapping)
            return f'(N-1 {left_str}{right_str})'
        else:
            return f'(T-1 {org_tokens[indices_mapping[root.i]]})'

class GenerativeTreeInducer(object):
    def __init__(self, 
                 model,  
                 beam_searcher, 
                 dataloader,
                 tokenizer,
                 device,
                 index=0):
        self._beam_searcher = beam_searcher
        self.model = model
        self.dataloader = dataloader
        self.tokenizer = tokenizer
        self._sep_word = ' '
        self.device = device
        self.index = index

    def induce(self, data_mode, output_dir="tmp/", format="nltk"):
        self.model.eval()
        trees = []
        data_iterator = tqdm(self.dataloader, desc="Iteration")
        with torch.no_grad():
            for _, inputs in enumerate(data_iterator):
                for k, v in inputs.items():
                    if v is not None and isinstance(v, torch.Tensor):
                        inputs[k] = v.to(self.device)
                
                # TODO: updata target_ids and target masks
                states = self._beam_searcher.beam_search(target_ids=inputs["chunk_input_ids"], 
                                                         target_masks=(inputs["chunk_masks"]>0).long(),
                                                         atom_spans=inputs["atom_spans"]) 
                
                for sent_id in range(inputs["masks"].shape[0]):
                    seq_len = inputs["masks"][sent_id].sum()
                    input_ids = inputs['input_ids'][sent_id, :seq_len].cpu().data.numpy()
                    root = states[sent_id][0].stack_top
                    tokens = ' '.join([convert_token(item) for item in self.tokenizer.convert_ids_to_tokens(input_ids)]).split()
                    # gold: 
                    # ['Skipper', "'s", 'Inc.', 'Bellevue', 'Wash.', 'said', 'it', 'signed', 'a', 'definitive', 'merger', 'agreement', 'for', 'a', 'National', 'Pizza', 'Corp.', 'unit', 'to', 'acquire', 'the', '90.6', '%', 'of', 'Skipper', "'s", 'Inc.', 'it', 'does', "n't", 'own', 'for', '11.50', 'a', 'share', 'or', 'about', '28.1', 'million']
                    sentence, spans = get_sentence_from_words(tokens, self._sep_word)
                    # logger.info(f'sentence: {sentence}')
                    outputs = self.tokenizer.encode_plus(sentence,
                                                          add_special_tokens=False,
                                                          return_offsets_mapping=True)
                    offset_mapping = outputs['offset_mapping']
                    word_starts, word_ends = align_spans(spans, offset_mapping)
                    atom_spans = []
                    indices_mapping = [0] * len(outputs['input_ids'])
                    for pos, (st, ed) in enumerate(zip(word_starts, word_ends)):
                        if ed > st:
                            atom_spans.append([st, ed])
                        for idx in range(st, ed + 1):
                            indices_mapping[idx] = pos
                    # print("tokens: ", tokens)
                    # print(f"root: {root}")
                    output1 = convert_to_ptb(root, tokens, atom_spans, indices_mapping)
                    if output1.startswith('(T-1'):
                        output1 = f'(NT-1 {output1})'
                    #output2 = convert_to_bracket(root, tokens, atom_spans, indices_mapping)
                    trees.append(output1)
        if format == "nltk":
            trees = [Tree.fromstring(t) for t in trees]
        return trees

