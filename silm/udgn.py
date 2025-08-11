from math import inf
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.init as init
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from silm import layers 

from transformers import PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import MaskedLMOutput

@dataclass(kw_only=True)
class MaskedLTLMOutput(MaskedLMOutput):
    """
    Base class for masked language models outputs.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Masked language modeling (MLM) loss.
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        heads (`torch.FloatTensor`): 
    """
    heads: Optional[torch.FloatTensor] = None
    tree_att_masks:  Optional[torch.FloatTensor] = None
    rels:  Optional[torch.FloatTensor] = None

class UDGNConfig(PretrainedConfig):
    
    model_type = "udgn"

    def __init__(self,
                 ntokens=1000,
                 emb_size=512,
                 n_dgn_layers=3,    # used to be nlayers
                 n_lstm_layers=3,   # used to be n_parser_layers
                 nhead=8,
                 head_size=128,
                 dropout=.1,
                 parser_dropout=.2,
                 dropatt=.0,
                 pos_emb=False,
                 pad=0,
                 detach_parser=False,
                 **kwargs):
        """Initialization.

        Args:
          ntokens: number of tokens in the embeddings
          emb_size: dimension of token embeddings
          n_dgn_layers: number of DGN layers
          n_lstm_layers: number of parsing layers (in the LSTM) 
          nhead: number of self-attention heads
          head_size: dimension of inputs and hidden states (in the DGN layer)
          dropout: dropout rate for the normal word embeddings
          parser_dropout: dropout rate for the parser embeddings
          dropatt: drop attention rate for the DGN layer
          pos_emb: bool, indicate whether use a learnable positional embedding
          pad: pad token index
          detach_parser: bool, indicate whether to detach the parser
        """
        self.ntokens = ntokens
        self.emb_size = emb_size
        self.n_dgn_layers = n_dgn_layers
        self.n_lstm_layers = n_lstm_layers
        self.nhead = nhead
        self.head_size = head_size
        self.dropout = dropout
        self.parser_dropout = parser_dropout
        self.dropatt = dropatt
        self.pos_emb = pos_emb
        self.pad = pad
        self.detach_parser = detach_parser
        super().__init__(**kwargs)


class UDGN(PreTrainedModel):
    """UDGN model."""
    config_class = UDGNConfig
    def __init__(self, config):


        super().__init__(config)

        self.drop = nn.Dropout(config.dropout)
        self.parser_drop = nn.Dropout(config.parser_dropout)

        self.emb        = nn.Embedding(config.ntokens, config.emb_size)
        self.parser_emb = nn.Embedding(config.ntokens, config.emb_size)
        if config.pos_emb:
            self.pos_emb = nn.Embedding(500, config.emb_size)

        self.layers = nn.ModuleList([
            layers.DGNLayer(
                config.emb_size, config.nhead, config.head_size,
                nrels=2, dropout=config.dropout, dropatt=config.dropatt)
            for _ in range(config.n_dgn_layers)])

        self.norm = nn.LayerNorm(config.emb_size)

        self.output_layer = nn.Linear(config.emb_size, config.ntokens)
        self.output_layer.weight = self.emb.weight

        self.parser_layers = nn.LSTM(config.emb_size, 
                                     config.emb_size, 
                                     config.n_lstm_layers,
                                     dropout=config.parser_dropout, 
                                     batch_first=True, 
                                     bidirectional=True)

        self.parser_ff = nn.Linear(config.emb_size * 2, config.emb_size * 2)

        self.pad = config.pad
        self.config = config

        self.criterion = nn.CrossEntropyLoss()

        self.init_weights()

    def init_weights(self):
        """Initialize token embedding and output bias."""
        initrange = 0.1
        self.emb.weight.data.uniform_(-initrange, initrange)
        self.parser_emb.weight.data.uniform_(-initrange, initrange)
        if hasattr(self, 'pos_emb'):
            self.pos_emb.weight.data.uniform_(-initrange, initrange)
        self.output_layer.bias.data.fill_(0)

        init.xavier_uniform_(self.parser_ff.weight)
        init.zeros_(self.parser_ff.bias)

    def parser_parameters(self):
        params = []
        params.extend(self.parser_emb.parameters())
        params.extend(self.parser_layers.parameters())
        params.extend(self.parser_ff.parameters())
        return params

    def lm_parameters(self):
        params = []
        params.extend(self.output_layer.parameters())
        params.extend(self.layers.parameters())
        params.extend(self.norm.parameters())
        return params

    def get_input_embeddings(self, kind="lm"):
        if kind=="lm":
            return self.emb
        elif kind=="parser":
            return self.parser_emb
    
    def visibility(self, x):
        """Mask pad tokens."""
        visibility = (x != self.pad)
        visibility = visibility[:, None, :].expand(-1, x.size(1), -1)
        return visibility

    def parse(self, x, deps=None):
        """Parse input sentence.

        Args:
          x: input tokens (required).
          pos: position for each token (optional).
        Returns:
          distance: syntactic distance
          height: syntactic height
        """

        if deps is not None:
            bsz, length = x.size()
            p = torch.zeros((bsz, length, length), device=x.device)
            deps = deps.clamp(min=0)
            p.scatter_(2, deps[:, :, None], 1)
            return p, p.log()

        mask = (x != self.pad)
        lengths = mask.sum(1).cpu().int()
        visibility = mask[:, None, :].expand(-1, x.size(1), -1)

        emb = self.parser_emb(x) # batch_size x sent_len -> batch_size x sent_len x emb_size

        h = self.parser_drop(emb) # dropout
        h = pack_padded_sequence(
            h, lengths.cpu(), batch_first=True, enforce_sorted=False)
        h, _ = self.parser_layers(h) # apply LSTM (Eq. 1 in paper). Usually Bidirectional -> double size of last dim
        h_padded, _ = pad_packed_sequence(h, batch_first=True, total_length=x.shape[1]) #, ])

        h = self.parser_drop(h_padded) # dropout. h: batch_size x sent_len x (emb_size * 2)
        # parent, child: both batch_size x sent_len x emb_size
        parent, child = self.parser_ff(h).chunk(2, dim=-1) # Eq. 2, 3

        scaling = self.config.emb_size ** -0.5 # D
        logits = torch.bmm(child, parent.transpose(1, 2)) * scaling # Eq. 4. batch_size x sent_len x sent_len
        logits = logits.masked_fill(~visibility, -inf)
        p = torch.softmax(logits, dim=-1) # Eq. 5. 

        return p, logits

    def generate_mask(self, p):
        """Compute head and cibling distribution for each token."""

        bsz, length, _ = p.size()

        eye = torch.eye(length, device=p.device, dtype=torch.bool)
        eye = eye[None, :, :].expand((bsz, -1, -1)) # batch_size x sent_len x sent_len
        head = p.masked_fill(eye, 0) # p=0 to be head of yourself
        child = head.transpose(1, 2)

        # Eq. 6: att_mask[i,j]: p that either i is head of j or j is head of i
        att_mask = head + child - head * child
        
        ones = torch.ones_like(p)

        left = ones.tril(-1) # left, right: batch_size x sent_len x sent_len
        right = ones.triu(1) # left: lower triangular, right: upper triangular (exclude diagonal)
        rels = torch.stack([left, right], dim=-1) # batch_size x sent_len x sent_len x 2

        if self.config.detach_parser:
            att_mask = att_mask.detach()

        return att_mask, head, rels

    def encode(self, x, pos, att_mask, rels):
        """UDGN encoding process."""
        att_mask = (att_mask + 1e-6).log()
        visibility = self.visibility(x)
        h = self.emb(x) # batch_size x sent_len -> batch_size x sent_len x emb_size
        if hasattr(self, 'pos_emb'):
            assert pos.max() < 500
            h = h + self.pos_emb(pos)
        h = self.drop(h)
        all_layers = [h]
        all_attn = []
        for i in range(self.config.n_dgn_layers):
            h, a = self.layers[i](
                h, rels, attn_mask=att_mask,
                key_padding_mask=visibility
            )
            all_layers.append(h) # h: batch_size x sent_len x emb_size
            all_attn.append(a) # a: 

        return h, all_layers, all_attn

    def forward(self, input_ids, attention_mask, pos=None, deps=None, labels=None, token_type_ids=None):
        """Pass the input through the encoder layer.

        Args:
          input_ids: input tokens (required).
          attention_mask: 1 where there is "real" input, 0 at the right where there is padding
          pos: position for each token (optional).
        Returns:
          loss: loss value
          state_dict: parsing results and raw output
        """

        batch_size, length = input_ids.size()
        # p: batch_size x sent length x sent length
        # p[i,j]: Probability that word i is dependent of j
        # logp: Same dims, but raw logits (not log probs!)
        p, logp = self.parse(input_ids, deps) 
        att_mask, head, rels = self.generate_mask(p)
        # att_mask, haed: batch_size x sent length x sent length. 
        # rels: batch_size x sent length x sent length x 2

        # apply DGN layers
        raw_output, all_layers, all_attn = self.encode(input_ids, pos, att_mask, rels)
        # raw_output: batch_size x sent length x emb_size
        # all_layers: list with num_layers elements: batch_size x sent length x emb_size
        # all_attn: list with num_layers elements: batch_size x sent length x sent length x nhead
        raw_output = self.norm(raw_output) # layer norm
        raw_output = self.drop(raw_output) # dropout
        logits = self.output_layer(raw_output)

        loss=None
        if labels is not None:
            # Restrict the gold the following cases here: 
            # a. Token can be viewed at through the attention mask
            # b. Token is masked 
            # c. Token is not padded
            target_mask = labels != -100
            #output = self.output_layer(raw_output[target_mask])

            # copied this part from roberta source code to match the shapes
            loss = self.criterion(logits.view(-1, self.config.ntokens), labels.view(-1))
            
        else:
            target_mask = attention_mask == self.pad
            #logits = self.output_layer(raw_output[target_mask])
            #loss = self.criterion(logits, attention_mask[target_mask])
        """
        return loss, \
            {'raw_output': raw_output,                       # batch_size x sent length x emb_size
             'att_mask': att_mask,                           # batch_size x sent length x sent length
             'head': head,                                   # batch_size x sent length x sent length
             'loghead': logp.view(batch_size * length, -1),  
             'all_layers': all_layers,                       # list: batch_size x sent length x emb_size
             'all_attn': all_attn}                           # list: batch_size x sent length x sent length x nhead
        """
        return MaskedLTLMOutput(
            loss=loss, 
            logits=logits, 
            heads=head, 
            tree_att_masks=att_mask,
            rels=rels
        )
