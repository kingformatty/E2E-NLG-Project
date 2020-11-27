
import copy
import torch.nn as nn
import torch.nn.functional as F
from components.model.modules import clones, LayerNorm, SublayerConnection
from components.model.modules.attention import MultiHeadedAttention
from components.model.modules import PositionwiseFeedForward, PositionalEncoding, Embeddings

class Decoder(nn.Module):
    "Generic N layer decoder with masking."
    def __init__(self, pre_net, layer, N, generator):
        super(Decoder, self).__init__()
        self.preNet = pre_net
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        self.generator = generator
        
    def forward(self, x, memory, src_mask, tgt_mask):
        x = self.preNet(x)
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        x = self.norm(x)
        return self.generator(x)

class Generator(nn.Module):
    "Define standard linear + softmax generation step."
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)

class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)
 
    def forward(self, x, memory, src_mask, tgt_mask):
        "Follow Figure 1 (right) for connections."
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)

def make_decoder(decoder_params, tgt_vocab):
   
    d_model = decoder_params["hidden_size"]
    input_size = decoder_params["input_size"]
    #assert d_model == encoder_params["input_size"]
    N = decoder_params["num_layers"]
    dropout = decoder_params["dropout"]
    h = decoder_params["n_head"]
    d_ff = decoder_params["d_ff"]

    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)

    model = Decoder(
            nn.Sequential(nn.Linear(input_size, d_model), nn.ReLU(), c(position)),
            DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N,
            Generator(d_model, tgt_vocab))

    # This was important from their code. 
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model

