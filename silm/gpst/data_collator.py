from typing import List, Union, Any, Dict
import torch
from transformers import DataCollatorForLanguageModeling
from transformers.data.data_collator import DataCollatorMixin
import numpy as np
from collections import OrderedDict
from silm.gpst import cppbackend
import codecs


class GPSTDataCollator(DataCollatorMixin):
    
    return_tensors: str = "pt"
    
    def __init__(self, tokenizer, ctx_size=96):
        self.lm_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.0)
        self.tokenizer = tokenizer
        self.ctx_size = ctx_size

    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]):
        batch = self.lm_collator(examples)

        """
        - `input_ids`: token ids, with padding id 0, such that 2048 tokens are reached (example shape: 66x81) for 66 sentences with max length 81
        - `chunk_input_ids`: The same tokens, but without padding
        - `chunk_masks`: Shape 2x1024, and in each row values 1-something to indicate the sentence in `input_ids`. Starting with 1!
        - `masks`: Same shape as the input ids, 1 for input tokens, 0 for padding
        - `group_ids`: One-dim with length 66 (num sents), values 0 and 1 to indicate which sentence goes in which row of the `chunk_input_ids` and `chunk_masks`
        - `atom_spans` is None, `span_ids` is an empty list, `external_vocab_ids` is None
        """
        assert batch["input_ids"].shape[1] == self.ctx_size, f'{batch["input_ids"].shape[1]}, {self.ctx_size}'
        max_batch_len = batch["attention_mask"].sum(dim=1).max().item() # old for formal language, depends on pad_token_id==0 batch["input_ids"].nonzero()[:,1].max()
        #chunk_input_ids = batch["input_ids"].view(-1) # old version, removes padding
        #chunk_input_ids = chunk_input_ids[chunk_input_ids != 0].unsqueeze(0)
        input_ids = batch["input_ids"][:,:max_batch_len]            
        attention_mask = batch["attention_mask"][:,:max_batch_len]
        chunk_input_ids = input_ids # when the assertion is true
        #group_ids = model_inputs["input_ids"].nonzero()[:,0]
        group_ids = torch.arange(input_ids.shape[0], dtype=int)
        #chunk_masks = (batch["input_ids"].nonzero()[:,0].unsqueeze(0))+1
        chunk_masks = attention_mask * (group_ids+1).unsqueeze(1)

        gpst_batch = {
            "input_ids": input_ids,        # batch_size (sents) x max_seq_length
            "masks": attention_mask,       
            "chunk_input_ids": chunk_input_ids,     # 
            "group_ids": group_ids.numpy(),
            "chunk_masks": chunk_masks
        }
        return gpst_batch

def load_span_tokenizer(external_vocab_path):
    
    external_vocab = []
    max_vocab_id = -1
    with codecs.open(external_vocab_path, mode='r') as f:
        for line in f:
            if '\t' in line:
                line = line.split('\t')[0]
            ids = line.split(',')
            ids = list(map(lambda x: int(x), ids))
            ids = np.array(ids)
            max_vocab_id = max(max_vocab_id, max(ids))
            external_vocab.append(ids)
    return cppbackend.SpanTokenizer(external_vocab, max_vocab_id + 1)

class SpanTokenizingSession:
    def __init__(self, span_tokenizer):
        self._span_tokenizer = span_tokenizer
        # 0 is reverved for no extra id
        self._span_indices = OrderedDict()
        self.external_vocab_idx = 1

    def tokenize(self, ids):
        results = self._span_tokenizer.tokenize(ids)
        span_idx = np.zeros((len(results),), dtype=np.int32)
        if len(results) > 0:
            assert len(results) % 3 == 0
            for group_id in range(len(results) // 3):
                idx, span_len, span_id = results[group_id * 3: group_id * 3 + 3]
                assert span_id >= 0
                span_idx[group_id * 3] = idx - span_len + 1
                span_idx[group_id * 3 + 1] = idx
                if span_id + 1 not in self._span_indices:
                    self._span_indices[span_id + 1] = self.external_vocab_idx
                    self.external_vocab_idx += 1
                span_idx[group_id * 3 + 2] = self._span_indices[span_id + 1]
        return span_idx

    @property
    def span_indices(self):
        return torch.tensor(np.array([0] + list(self._span_indices.keys())))

def _find_point_in_spans(point, start_index, spans):
    """

        Find which subword token a point in the original sentence lies in.

    Args:
            point(int): an index in the sentence string
            start_index(int): the index of the subword in `spans`
                                to start searching
            spans(list[tuple[int]]): huggingface tokenizers' offset_mapping
                                        each subword's index span in the sentence string


    Returns:    
                index(int): the index of the subword in `spans`
                                that `point` belongs to

    """
    index = start_index
    while index < len(spans):
        span = spans[index]
        if span is not None and span[0] < span[1]:  # span is not empty
            if point >= span[0] and point < span[1]:
                break
        else:
            assert span is None or span[0] == span[1] == 0
        index += 1
    return index


def align_spans(original_spans, token_spans):
    """
    
    Map each word to its subtokens.

    Args:   original_spans(list[tuple[int]]): slice indices to index each word 
                                                in the sentence string (word_sep considered)
            token_spans(list[tuple[int]]): huggingface tokenizers' offset_mapping
                                                each subword's index span in the sentence string
    Returns:
            word_starts(list[int]): word_starts[i]=j means ith word begins at jth subword
            word_ends(list[int]): word_ends[i]=j means ith word ends in jth subword (inclusive)
    """
    word_starts = []
    word_ends = []

    while token_spans and (token_spans[-1] is None or token_spans[-1][1] == 0):
        token_spans.pop()  # remove trailing empty spans

    last = 0
    for (start, end) in original_spans:
        first = _find_point_in_spans(start, last, token_spans)
        last = _find_point_in_spans(end - 1, first, token_spans)

        word_starts.append(first)
        word_ends.append(last)

    return word_starts, word_ends

class DefaultCollator:
    def __init__(self, enable_group=True, external_vocab_path=None):
        # enable_group is deprecated

        if external_vocab_path is not None:
            self.span_tokenizer = load_span_tokenizer(external_vocab_path)
        else:
            self.span_tokenizer = None

    def generative_r2d2_collate_fn(self, input_list) -> Dict[str, torch.Tensor]:
        '''
            input_list: [{"text": ..., "sentence_splits":...},...]
        '''
        ids_list = []
        group_ids = []
        max_sent_len = 0
        chunk_ids_list = []
        # chunk_masks = []
        segment_ids_list = []
        span_indices = []
        max_input_len = max(map(lambda x: len(x['text']), input_list))
        

        for sent_id, item in enumerate(input_list):
            chunk_ids_list.append(item['text'])
            chunk_size = len(item['text'])
            # chunk_mask = np.zeros( (max_input_len, max_input_len) )
            segment_ids = np.zeros(max_input_len)
            segment_ids_list.append(segment_ids)
            # chunk_masks.append(chunk_mask)
            splits = item['sentence_splits']
            splits.append(chunk_size)

            prev_idx = 0
            # cppbackend.create_mask(chunk_mask, np.array(splits))
            for segment_id, split_idx in enumerate(splits):
                if split_idx > prev_idx:
                    ids_segment = item['text'][prev_idx: split_idx]
                    if self.span_tokenizer is not None:
                        results = self.span_tokenizer.tokenize(ids_segment)
                        span_idx = np.zeros((len(results),))
                        if len(results) > 0:
                            assert len(results) % 3 == 0
                            for group_id in range(len(results) // 3):
                                idx, span_len, span_id = results[group_id * 3: group_id * 3 + 3]
                                span_idx[group_id * 3] = idx - span_len + 1
                                span_idx[group_id * 3 + 1] = idx
                                span_idx[group_id * 3 + 2] = span_id
                                
                        span_indices.append(span_idx)
                    # ids_lens = np.floor(np.log10(ids_segment)) + 1
                    # splits = np.cumsum(np.array(list(ids_lens)) + 1) - 1
                    # target = ','.join([f'{id}' for id in tgt_ids])
                    ids_list.append(ids_segment)
                    # chunk_mask[prev_idx: split_idx, prev_idx: split_idx].fill(1)
                    # print(segment_id)
                    segment_ids[prev_idx: split_idx].fill(segment_id + 1)
                    group_ids.append(sent_id)
                    max_sent_len = max(max_sent_len, split_idx - prev_idx)
                prev_idx = split_idx

        # print(chunk_mask)
        # segment_ids = torch.tensor(segment_ids)
        # print(segment_ids)
        
        # padding
        masks = []
        for sent_i in range(len(ids_list)):
            pad_len = max_sent_len - len(ids_list[sent_i])
            masks.append(np.array([1] * len(ids_list[sent_i]) + [0] * pad_len, dtype=np.int32))
            # print(ids_list[sent_i])
            ids_list[sent_i] = np.append(np.array(ids_list[sent_i], dtype=np.int32), np.array([0] * pad_len))

        for chunk_id, chunk_ids in enumerate(chunk_ids_list):
            pad_len = max_input_len - len(chunk_ids)
            chunk_ids_list[chunk_id] = np.append(chunk_ids, np.array([-100] * pad_len))

        return {"chunk_input_ids": torch.tensor(np.array(chunk_ids_list, dtype=np.int32), dtype=torch.long),
                "chunk_masks": torch.tensor(np.array(segment_ids_list, dtype=np.int32), dtype=torch.long),
                "input_ids": torch.tensor(np.array(ids_list, dtype=np.int32), dtype=torch.long), 
                "masks": torch.tensor(np.array(masks, dtype=np.int32), dtype=torch.long), 
                "group_ids": np.array(group_ids),
                "span_ids": span_indices}

    def generative_r2d2_collate_fn_ext(self, input_list) -> Dict[str, torch.Tensor]:
        '''
            input_list: [{"text": ..., "sentence_splits":...},...]
        '''
        ids_list = []
        group_ids = []
        max_sent_len = 0
        chunk_ids_list = []
        # chunk_masks = []
        span_indices = []
        max_input_len = max(map(lambda x: len(x['text']), input_list))
        segment_ids_list = []
        external_dict = OrderedDict()
        external_vocab_idx = 1  # start from 1, 0 is reserved for empty span ids
        tokenizer_session = SpanTokenizingSession(self.span_tokenizer)
        for sent_id, item in enumerate(input_list):
            chunk_ids_list.append(item['text'])
            chunk_size = len(item['text'])
            # chunk_mask = np.zeros( (max_input_len, max_input_len) )
            # chunk_masks.append(chunk_mask)
            segment_ids = np.zeros(max_input_len)
            segment_ids_list.append(segment_ids)
            splits = item['sentence_splits']
            splits.append(chunk_size)

            prev_idx = 0
            # cppbackend.create_mask(chunk_mask, np.array(splits))
            for segment_id, split_idx in enumerate(splits):
                if split_idx > prev_idx:
                    ids_segment = item['text'][prev_idx: split_idx]
                    if self.span_tokenizer is not None:
                        span_idx = tokenizer_session.tokenize(ids_segment)
                        span_indices.append(span_idx)

                    ids_list.append(ids_segment)
                    # chunk_mask[prev_idx: split_idx, prev_idx: split_idx].fill(1)
                    segment_ids[prev_idx: split_idx].fill(segment_id + 1)
                    group_ids.append(sent_id)
                    max_sent_len = max(max_sent_len, split_idx - prev_idx)
                prev_idx = split_idx
        
        # print(chunk_mask)
        # segment_ids = torch.tensor(segment_ids)
        # print(segment_ids)
        # padding
        masks = []
        for sent_i in range(len(ids_list)):
            pad_len = max_sent_len - len(ids_list[sent_i])
            masks.append(np.array([1] * len(ids_list[sent_i]) + [0] * pad_len, dtype=np.int32))
            # print(ids_list[sent_i])
            ids_list[sent_i] = np.append(np.array(ids_list[sent_i], dtype=np.int32), np.array([0] * pad_len))
        
        for chunk_id, chunk_ids in enumerate(chunk_ids_list):
            pad_len = max_input_len - len(chunk_ids)
            chunk_ids_list[chunk_id] = np.append(chunk_ids, np.array([-100] * pad_len))
        
        if self.span_tokenizer is not None:
            external_ids = tokenizer_session.span_indices
        else:
            external_ids = None
                # "chunk_input_ids": torch.tensor(np.array()),
                # "chunk_masks": torch.tensor(),
        return {"chunk_input_ids": torch.tensor(np.array(chunk_ids_list, dtype=np.int32), dtype=torch.long),
                "chunk_masks": torch.tensor(np.array(segment_ids_list, dtype=np.int32), dtype=torch.long),
                "input_ids": torch.tensor(np.array(ids_list, dtype=np.int32), dtype=torch.long), 
                "masks": torch.tensor(np.array(masks, dtype=np.int32), dtype=torch.long), 
                "group_ids": np.array(group_ids),
                "span_ids": span_indices,
                "external_vocab_ids": external_ids}

class TextCollator(DefaultCollator):
    def __init__(self, tokenizer, splitter, external_vocab_path=None, add_special_tokens=True):
        self.tokenizer = tokenizer
        self.splitter = splitter
        self.add_special_tokens = add_special_tokens
        super().__init__(external_vocab_path=external_vocab_path)

    def collate_fn(self, input_list) -> Dict[str, torch.Tensor]:
        input_ids_list = []
        atom_spans_batch = []
        for sentence in input_list:
            tokens, split_word = self.splitter(sentence)
            offset = 0
            spans = []
            for word in tokens:
                length = len(word)
                spans.append((offset, offset + length))
                offset += length + len(split_word)
            outputs = self.tokenizer.encode_plus(sentence,
                                                 add_special_tokens=self.add_special_tokens,
                                                 return_offsets_mapping=True)
            input_ids = outputs['input_ids']
            offset_mapping = outputs['offset_mapping']
            word_starts, word_ends = align_spans(spans, offset_mapping)
            atom_spans = [] # minimal span should be a whole word
            for pos, (st, ed) in enumerate(zip(word_starts, word_ends)):
                if ed > st:
                    atom_spans.append([st, ed])
            input_ids_list.append({'text':input_ids, 'sentence_splits': []})
            atom_spans_batch.append(atom_spans)

        out_dict = self.generative_r2d2_collate_fn_ext(input_ids_list)
        out_dict['atom_spans'] = atom_spans_batch
        return out_dict
