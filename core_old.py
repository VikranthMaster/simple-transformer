import math
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from tokenizer import *
from embedding import *
def prepare_input(token_ids, embedding_layer, dropout_p: float = 0.1, training:bool = False):
    x = embedding_layer(token_ids)
    seq_len, d_model = x.size(1), x.size(2)
    pe = positional_encoding(seq_len=seq_len, d_model=d_model, device=x.device)
    x = x+pe
    return nn.functional.dropout(x, p=dropout_p, training=training)

def scaled_dot_product(q,k,v, mask=None):
    """
    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
    """
    d_k = k.size(-1)
    scores= torch.matmul(q,k.transpose(-2,-1))/ math.sqrt(d_k)

    if mask != None:
        scores = scores.masked_fill(mask == False, float("-inf"))

    attn = torch.softmax(scores, dim=-1)
    output = torch.matmul(attn,v)
    return output, attn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads


        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)


        self.dropout = nn.Dropout(dropout)


    def split_heads(self,x):
        batch_size, seq_len, _ = x.shape
        x = x.view(batch_size, seq_len, self.num_heads, self.d_k)
        return x.transpose(1,2)

    def combine_heads(self,x):
        batch_size, _, seq_len, _ = x.shape
        x = x.transpose(1,2).contiguous()
        return x.view(batch_size, seq_len, self.d_model)

    def forward(self, query,key, value,mask=None):
        q  = self.split_heads(self.w_q(query))
        k  = self.split_heads(self.w_k(key))
        v  = self.split_heads(self.w_v(value))
        attn_output, attn_weights = scaled_dot_product(q,k,v,mask)
        combined = self.combine_heads(attn_output)
        output = self.w_o(combined)
        return self.dropout(output), attn_weights

class FeedForwardNN(nn.Module):
    def __init__(self, d_model, d_ff = 1024, dropout = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.linear2(self.dropout(torch.relu(self.linear1(x))))
    
class AddNorm(nn.Module):
    def __init__(self, d_model, dropout = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self,x, sublayer_output):
        return self.norm(x + self.dropout(sublayer_output))
    

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff = 2048, dropout = 0.1):
        super().__init__()
        self.attn = MultiHeadAttention(d_model=d_model, num_heads=num_heads, dropout=dropout)
        self.ffn = FeedForwardNN(d_model=d_model, d_ff=d_ff, dropout=dropout)
        self.add_norm1 = AddNorm(d_model=d_model, dropout=dropout)
        self.add_norm2 = AddNorm(d_model=d_model, dropout=dropout)

    def forward(self,x, src_mask=None):
        attn_output, _ = self.attn(x,x,x, src_mask)
        x = self.add_norm1(x, attn_output)

        ffn_output = self.ffn(x)
        x = self.add_norm2(x, ffn_output)

        return x

class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model=512, num_heads = 8, d_ff = 2048, num_layers = 6, dropout = 0.1):
        super().__init__()
        self.embedding = InputEmbeddings(vocab_size=vocab_size, d_model=d_model)
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([ #ModuleList is simply a list that PyTorch understands contains neural network layers.
            EncoderLayer(d_model=d_model, num_heads=num_heads, d_ff=d_ff, dropout=dropout)
            for _ in range(num_layers)
        ])


    def forward(self, src_token_ids, src_mask = None):
        x = self.embedding(src_token_ids)
        pe = positional_encoding(x.size(1), x.size(2), x.device)
        x = self.dropout(x + pe)

        for layer in self.layers:
            x = layer(x, src_mask)

        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff=2048, dropout = 0.1):
        super().__init__()
        self.attn = MultiHeadAttention(d_model=d_model, num_heads=num_heads, dropout=dropout)
        self.cross_attn = MultiHeadAttention(d_model=d_model, num_heads=num_heads, dropout=dropout)
        self.ffn = FeedForwardNN(d_model=d_model, d_ff=d_ff, dropout=dropout)

        self.add_norm1 = AddNorm(d_model=d_model, dropout=dropout)
        self.add_norm2 = AddNorm(d_model=d_model, dropout=dropout)
        self.add_norm3 = AddNorm(d_model=d_model, dropout=dropout)

    def forward(self, x, encoder_output, target_mask= None, src_mask=None):
        attn_output, _ = self.attn(x,x,x,target_mask)
        x = self.add_norm1(x, attn_output)

        cross_attn_output, cross_attn_weights = self.cross_attn(
            query = x, key = encoder_output, value = encoder_output, mask=src_mask
        )
        x = self.add_norm2(x, cross_attn_output)


        ffn_output = self.ffn(x)
        x = self.add_norm3(x, ffn_output)

        return x, cross_attn_weights

class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model=512, num_heads=8, d_ff =2048, num_layers=6, dropout=0.1):
        super().__init__()
        self.embedding = InputEmbeddings(vocab_size=vocab_size, d_model=d_model)
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])


    def forward(self, target_token_ids, encoder_output, target_mask = None, src_mask=None):
        x = self.embedding(target_token_ids)
        pe = positional_encoding(x.shape[1], x.shape[2], device=x.device)
        x = self.dropout(x+pe)

        attn_weights = None
        for layer in self.layers:
            x,attn_weights = layer(x, encoder_output, target_mask, src_mask)

        return x, attn_weights