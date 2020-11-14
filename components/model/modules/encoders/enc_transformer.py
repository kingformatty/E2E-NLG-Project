
import copy
import torch.nn as nn
from components.model.modules import clones, LayerNorm, SublayerConnection
from components.model.modules.attention import MultiHeadedAttention
from components.model.modules import PositionwiseFeedForward, PositionalEncoding

class Encoder(nn.Module):
    "Core encoder is a stack of N layers"
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        self.hidden_size = self.layers[0].size
        
    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)

def make_encoder(encoder_params):
   
    d_model = encoder_params["hidden_size"]
    assert d_model == encoder_params["input_size"]
    N = encoder_params["num_layers"]
    dropout = encoder_params["dropout"]
    h = encoder_params["n_head"]
    d_ff = encoder_params["d_ff"]

    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    model = Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N)
    
    # This was important from their code. 
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model

