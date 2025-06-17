#pylint: disable=E1101
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class create_classifier(nn.Module):
 
    def __init__(self, latent_dim, nhidden=16, N=2):
        super(create_classifier, self).__init__()
        self.gru_rnn = nn.GRU(latent_dim, nhidden, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(nhidden, 300),
            nn.ReLU(),
            nn.Linear(300, 300),
            nn.ReLU(),
            nn.Linear(300, N))
       
    def forward(self, z):
        _, out = self.gru_rnn(z)
        return self.classifier(out.squeeze(0))
    

class multiTimeAttention(nn.Module):
    
    def __init__(self, input_dim, nhidden=16, 
                 embed_time=16, num_heads=1):
        super(multiTimeAttention, self).__init__()
        assert embed_time % num_heads == 0
        self.embed_time = embed_time
        self.embed_time_k = embed_time // num_heads
        self.h = num_heads
        self.dim = input_dim
        self.nhidden = nhidden
        self.linears = nn.ModuleList([nn.Linear(embed_time, embed_time), 
                                      nn.Linear(embed_time, embed_time),
                                      nn.Linear(input_dim*num_heads, nhidden)])
 
    def attention(self, query, key, value, mask=None, query_mask=None, key_mask=None, dropout=None):
        """Compute 'Scaled Dot Product Attention'
        
        Changes made:
        - attention() now accepts `query_mask` and `key_mask` as inputs.
        - Both these inputs are meant to be used to define masks for query times,
            it's just that query_mask defines the mask for the 2D `query_times` for
            the encoder, but `key_mask` defines the mask for `query_times` for the
            decoder, because in the decoder the query times becomes the key.
        """
        dim = value.size(-1)
        d_k = query.size(-1)
        #print(query.shape, key.shape)
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(d_k)
        scores = scores.unsqueeze(-1).repeat_interleave(dim, dim=-1)
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(-3) == 0, -1e9)

        # Mask out scores where query_mask == 0.
        if query_mask is not None:
            B, h, Tq, Tk, d = scores.shape
            # reshape to [B, 1, Tq, 1, 1] then expand to [B, h, Tq, Tk, dim]
            qm = query_mask.view(B, 1, Tq, 1, 1).expand(B, h, Tq, Tk, dim)
            scores = scores.masked_fill(qm == 0, -1e9)
        # Mask out scores where key_mask = 0.
        if key_mask is not None:
            B, h, Tq, Tk, d = scores.shape
            km = key_mask.view(B, 1, Tk, 1, 1).expand(B, h, Tk, Tq, dim).transpose(2, 3)
            scores = scores.masked_fill(km == 0, -1e9)

        p_attn = F.softmax(scores, dim = -2)
        if dropout is not None:
            p_attn = dropout(p_attn)
        return torch.sum(p_attn*value.unsqueeze(-3), -2), p_attn
 

    def forward(self, query, key, value, mask=None, query_mask=None, key_mask=None, dropout=None, return_att=False):
        "Compute 'Scaled Dot Product Attention'"
        batch, seq_len, dim = value.size()
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        value = value.unsqueeze(1)
        query, key = [l(x).view(x.size(0), -1, self.h, self.embed_time_k).transpose(1, 2)
                      for l, x in zip(self.linears, (query, key))]
        x, attn = self.attention(query, key, value, mask, query_mask, key_mask, dropout)
        x = x.transpose(1, 2).contiguous() \
             .view(batch, -1, self.h * dim)
        if return_att:
            return self.linears[-1](x), attn
        else:
            return self.linears[-1](x)
    
class enc_mtan_rnn(nn.Module):
    def __init__(self, input_dim, latent_dim=2, nhidden=16, 
                 embed_time=16, num_heads=1, learn_emb=False, device='cuda'):
        super(enc_mtan_rnn, self).__init__()
        self.embed_time = embed_time
        self.dim = input_dim
        self.device = device
        self.nhidden = nhidden
        #self.query = query
        self.learn_emb = learn_emb
        self.att = multiTimeAttention(2*input_dim, nhidden, embed_time, num_heads)
        self.gru_rnn = nn.GRU(nhidden, nhidden, bidirectional=True, batch_first=True)
        #self.conv = nn.Conv1d(256, nhidden, kernel_size=3)  # ADDED
        self.hiddens_to_z0 = nn.Sequential(
            nn.Linear(2*nhidden, 50),
            nn.ReLU(),
            nn.Linear(50, latent_dim * 2))
        if learn_emb:
            self.periodic = nn.Linear(1, embed_time-1)
            self.linear = nn.Linear(1, 1) 
            #self.linear = nn.Linear(1, embed_time)

    def learn_time_embedding(self, tt):
        tt = tt.to(self.device)
        tt = tt.to(torch.float32)
        tt = tt.unsqueeze(-1)
        #return self.linear(tt)
        out2 = torch.sin(self.periodic(tt))
        out1 = self.linear(tt)
        return torch.cat([out1, out2], -1)
    
    def fixed_time_embedding(self, pos):
        d_model=self.embed_time
        pe = torch.zeros(pos.shape[0], pos.shape[1], d_model)
        position = 48.*pos.unsqueeze(2)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(np.log(10.0) / d_model))
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe
       
    def forward(self, x, time_steps, query_times, return_att=False):
        time_steps = time_steps.cpu().to(torch.float32)
        mask = x[:, :, self.dim:]
        mask = torch.cat((mask, mask), 2)
        if self.learn_emb:
            # NOTE: Only when learn_emb is True, we have changed the original mTAN code
            # to accept two-dimensional query values.
            # First, a 2D query mask is created. The `unsqueeze(0)` operation in the line: `query = self.learn_time_embedding(..)`
            # is removed because the 2D matrix already include the batch dimension as the first dimension.
            query_mask = (query_times != -999).float()
            key = self.learn_time_embedding(time_steps).to(self.device)
            query = self.learn_time_embedding(query_times).to(self.device)
            #print(query.shape, query_times.shape, key.shape, time_steps.shape)
        else:
            key = self.fixed_time_embedding(time_steps).to(self.device)
            query = self.fixed_time_embedding(query_times.unsqueeze(0)).to(self.device)
        out = self.att(query, key, x.float(), mask.float(), query_mask.float(), return_att=return_att)
        if return_att:
            out = out[0]
        out, _ = self.gru_rnn(out)
        out = self.hiddens_to_z0(out)
        return out
    
    
class dec_mtan_rnn(nn.Module):
 
    def __init__(self, input_dim, latent_dim=2, nhidden=16, 
                 embed_time=16, num_heads=1, learn_emb=False, device='cuda'):
        super(dec_mtan_rnn, self).__init__()
        self.embed_time = embed_time
        self.dim = input_dim
        self.device = device
        self.nhidden = nhidden
        #self.query = query
        self.learn_emb = learn_emb
        self.att = multiTimeAttention(2*nhidden, 2*nhidden, embed_time, num_heads)
        self.gru_rnn = nn.GRU(latent_dim, nhidden, bidirectional=True, batch_first=True)    
        #self.conv = nn.Conv1d(latent_dim, nhidden, kernel_size=3)  # ADDED
        self.z0_to_obs = nn.Sequential(
            nn.Linear(2*nhidden, 50),
            nn.ReLU(),
            nn.Linear(50, input_dim))
        if learn_emb:
            self.periodic = nn.Linear(1, embed_time-1)
            self.linear = nn.Linear(1, 1)
        
        
    def learn_time_embedding(self, tt):
        tt = tt.to(self.device)
        tt = tt.to(torch.float32)
        tt = tt.unsqueeze(-1)
        out2 = torch.sin(self.periodic(tt))
        out1 = self.linear(tt)
        return torch.cat([out1, out2], -1)
        
        
    def fixed_time_embedding(self, pos):
        d_model = self.embed_time
        pe = torch.zeros(pos.shape[0], pos.shape[1], d_model)
        position = 48.*pos.unsqueeze(2)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(np.log(10.0) / d_model))
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe
       
    def forward(self, z, time_steps, query_times):
        out, _ = self.gru_rnn(z)
        time_steps = time_steps.cpu()
        if self.learn_emb:
            # NOTE: Only when learn_emb is True, we have changed the original mTAN code (just like that for `enc_mtan_rnn`)
            # to accept two-dimensional query values.
            # First, a 2D key mask is created using the query times (since query becomes keys and vice versa in decoder compared to encoder).
            # The key and the corresponding key mask are repeated across the batch dimension.
            # The `unsqueeze(0)` operation in the line: `query = self.learn_time_embedding(..)`
            # is removed because the 2D matrix already include the batch dimension as the first dimension.
            query = self.learn_time_embedding(time_steps).to(self.device)
            key_mask = (query_times != -999).float()
            key = self.learn_time_embedding(query_times).to(self.device)
            b_times_nsamples = len(query) // len(key)
            key = key.repeat(b_times_nsamples, 1, 1)
            key_mask = key_mask.repeat(b_times_nsamples, 1)
            #print(key_mask.shape)
        else:
            query = self.fixed_time_embedding(time_steps).to(self.device)
            key = self.fixed_time_embedding(query_times.unsqueeze(0)).to(self.device)

        out = self.att(query, key, out, key_mask=key_mask)
        out = self.z0_to_obs(out)
        return out        
   
    
class enc_mtan_classif(nn.Module):
 
    def __init__(self, input_dim, query, nhidden=16, 
                 embed_time=16, num_heads=1, learn_emb=True, freq=10., device='cuda'):
        super(enc_mtan_classif, self).__init__()
        assert embed_time % num_heads == 0
        self.freq = freq
        self.embed_time = embed_time
        self.learn_emb = learn_emb
        self.dim = input_dim
        self.device = device
        self.nhidden = nhidden
        self.query = query
        self.att = multiTimeAttention(2*input_dim, nhidden, embed_time, num_heads)
        self.classifier = nn.Sequential(
            nn.Linear(nhidden, 300),
            nn.ReLU(),
            nn.Linear(300, 300),
            nn.ReLU(),
            nn.Linear(300, 2))
        self.enc = nn.GRU(nhidden, nhidden)
        if learn_emb:
            self.periodic = nn.Linear(1, embed_time-1)
            self.linear = nn.Linear(1, 1)
    
    
    def learn_time_embedding(self, tt):
        tt = tt.to(self.device)
        tt = tt.unsqueeze(-1)
        out2 = torch.sin(self.periodic(tt))
        out1 = self.linear(tt)
        return torch.cat([out1, out2], -1)
       
        
    def time_embedding(self, pos, d_model):
        pe = torch.zeros(pos.shape[0], pos.shape[1], d_model)
        position = 48.*pos.unsqueeze(2)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(np.log(self.freq) / d_model))
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe
    
       
    def forward(self, x, time_steps):
        time_steps = time_steps.cpu()
        mask = x[:, :, self.dim:]
        mask = torch.cat((mask, mask), 2)
        if self.learn_emb:
            key = self.learn_time_embedding(time_steps).to(self.device)
            query = self.learn_time_embedding(self.query.unsqueeze(0)).to(self.device)
        else:
            key = self.time_embedding(time_steps, self.embed_time).to(self.device)
            query = self.time_embedding(self.query.unsqueeze(0), self.embed_time).to(self.device)
            
        out = self.att(query, key, x, mask)
        out = out.permute(1, 0, 2)
        _, out = self.enc(out)
        return self.classifier(out.squeeze(0))




class enc_mtan_classif_activity(nn.Module):
 
    def __init__(self, input_dim, nhidden=16, 
                 embed_time=16, num_heads=1, learn_emb=True, freq=10., device='cuda'):
        super(enc_mtan_classif_activity, self).__init__()
        assert embed_time % num_heads == 0
        self.freq = freq
        self.embed_time = embed_time
        self.learn_emb = learn_emb
        self.dim = input_dim
        self.device = device
        self.nhidden = nhidden
        self.att = multiTimeAttention(2*input_dim, nhidden, embed_time, num_heads)
        self.gru = nn.GRU(nhidden, nhidden, batch_first=True)
        self.classifier = nn.Linear(nhidden, 11)
        if learn_emb:
            self.periodic = nn.Linear(1, embed_time-1)
            self.linear = nn.Linear(1, 1)
    
    
    def learn_time_embedding(self, tt):
        tt = tt.to(self.device)
        tt = tt.unsqueeze(-1)
        out2 = torch.sin(self.periodic(tt))
        out1 = self.linear(tt)
        return torch.cat([out1, out2], -1)
       
        
    def time_embedding(self, pos, d_model):
        pe = torch.zeros(pos.shape[0], pos.shape[1], d_model)
        position = 48.*pos.unsqueeze(2)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(np.log(self.freq) / d_model))
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe
    
       
    def forward(self, x, time_steps):
        batch = x.size(0)
        time_steps = time_steps.cpu()
        mask = x[:, :, self.dim:]
        mask = torch.cat((mask, mask), 2)
        if self.learn_emb:
            key = self.learn_time_embedding(time_steps).to(self.device)
        else:
            key = self.time_embedding(time_steps, self.embed_time).to(self.device)
        out = self.att(key, key, x, mask)
        out, _ = self.gru(out)
        out = self.classifier(out)
        return out
    
    
    
class enc_interp(nn.Module):
    def __init__(self, input_dim, query, latent_dim=2, nhidden=16, device='cuda'):
        super(enc_interp, self).__init__()
        self.dim = input_dim
        self.device = device
        self.nhidden = nhidden
        self.query = query
        self.cross = nn.Linear(2*input_dim, 2*input_dim)
        self.bandwidth = nn.Linear(1, 2*input_dim, bias=False)
        self.gru_rnn = nn.GRU(2*input_dim, nhidden, bidirectional=True, batch_first=True)
        self.hiddens_to_z0 = nn.Sequential(
            nn.Linear(2*nhidden, 50),
            nn.ReLU(),
            nn.Linear(50, latent_dim * 2))

    
    def attention(self, query, key, value, mask=None, dropout=None):
        "Compute 'Scaled Dot Product Attention'"
        query, key = query.to(self.device), key.to(self.device)
        batch, seq_len, dim = value.size()
        scores = -(query.unsqueeze(-1) - key.unsqueeze(-2))**2
        scores = scores[:, :, :, None].repeat(1, 1, 1, dim)
        bandwidth = torch.log(1 + torch.exp(self.bandwidth(torch.ones(1, 1, 1, 1).to(self.device))))
        scores = scores * bandwidth
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1) == 0, -1e9)
        p_attn = F.softmax(scores, dim = -2)
        if dropout is not None:
            p_attn = dropout(p_attn)
        #return torch.sum(p_attn*value, -2), p_attn
        return torch.sum(p_attn*value.unsqueeze(1), -2), p_attn
       
        
    def forward(self, x, time_steps):
        #time_steps = time_steps.cpu()
        mask = x[:, :, self.dim:]
        mask = torch.cat((mask, mask), 2)
        out, _ = self.attention(self.query.unsqueeze(0), time_steps, x, mask)
        out = self.cross(out)
        out, _ = self.gru_rnn(out)
        out = self.hiddens_to_z0(out)
        return out
    
    
class dec_interp(nn.Module):
 
    def __init__(self, input_dim, query, latent_dim=2, nhidden=16, device='cuda'):
        super(dec_interp, self).__init__()
        self.dim = input_dim
        self.device = device
        self.nhidden = nhidden
        self.query = query
        self.bandwidth = nn.Linear(1, 2*nhidden, bias=False)
        self.gru_rnn = nn.GRU(latent_dim, nhidden, bidirectional=True, batch_first=True)
        self.z0_to_obs = nn.Sequential(
            nn.Linear(2*nhidden, 50),
            nn.ReLU(),
            nn.Linear(50, input_dim))
        
    
    def attention(self, query, key, value, mask=None, dropout=None):
        "Compute 'Scaled Dot Product Attention'"
        query, key = query.to(self.device), key.to(self.device)
        batch, seq_len, dim = value.size()
        scores = -(query.unsqueeze(-1) - key.unsqueeze(-2))**2
        scores = scores[:, :, :, None].repeat(1, 1, 1, dim)
        bandwidth = torch.log(1 + torch.exp(self.bandwidth(torch.ones(1, 1, 1, 1).to(self.device))))
        scores = scores * bandwidth
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1) == 0, -1e9)
        p_attn = F.softmax(scores, dim = -2)
        if dropout is not None:
            p_attn = dropout(p_attn)
        #return torch.sum(p_attn*value, -2), p_attn
        return torch.sum(p_attn*value.unsqueeze(1), -2), p_attn
        
    def forward(self, z, time_steps):
        out, _ = self.gru_rnn(z)
        out, _ = self.attention(time_steps, self.query.unsqueeze(0), out)
        out = self.z0_to_obs(out)
        return out        

    
class enc_rnn3(nn.Module):
    def __init__(self, input_dim, query, latent_dim=2, nhidden=16, 
                 embed_time=16, use_classif=False, learn_emb=False, device='cuda'):
        super(enc_rnn3, self).__init__()
        self.use_classif = use_classif 
        self.embed_time = embed_time
        self.dim = input_dim
        self.device = device
        self.nhidden = nhidden
        self.query = query
        self.learn_emb = learn_emb
        self.cross = nn.Linear(2*input_dim, nhidden)
        if use_classif:
            self.gru_rnn = nn.GRU(nhidden, nhidden, batch_first=True)
        else:
            self.gru_rnn = nn.GRU(nhidden, nhidden, bidirectional=True, batch_first=True)
        self.linears = nn.ModuleList([nn.Linear(embed_time, embed_time) for _ in range(2)])
        if learn_emb:
            self.periodic = nn.Linear(1, embed_time-1)
            self.linear = nn.Linear(1, 1)
        if use_classif:
            self.classifier = nn.Sequential(
                nn.Linear(nhidden, 300),
                nn.ReLU(),
                nn.Linear(300, 300),
                nn.ReLU(),
                nn.Linear(300, 2))
        else:
            self.hiddens_to_z0 = nn.Sequential(
                nn.Linear(2*nhidden, 50),
                nn.ReLU(),
                nn.Linear(50, latent_dim * 2))
    
    def learn_time_embedding(self, tt):
        tt = tt.to(self.device)
        tt = tt.unsqueeze(-1)
        out2 = torch.sin(self.periodic(tt))
        out1 = self.linear(tt)
        return torch.cat([out1, out2], -1)
    
    
    def fixed_time_embedding(self, pos):
        d_model=self.embed_time
        pe = torch.zeros(pos.shape[0], pos.shape[1], d_model)
        position = 48.*pos.unsqueeze(2)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(np.log(10.0) / d_model))
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe
    
    def attention(self, query, key, value, mask=None, dropout=None):
        "Compute 'Scaled Dot Product Attention'"
        batch, seq_len, dim = value.size()
        d_k = query.size(-1)
        query, key = [l(x) for l, x in zip(self.linears, (query, key))]
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(d_k)
        scores = scores[:, :, :, None].repeat(1, 1, 1, dim)
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1) == 0, -1e9)
        p_attn = F.softmax(scores, dim = -2)
        if dropout is not None:
            p_attn = dropout(p_attn)
        #return torch.sum(p_attn*value, -2), p_attn
        return torch.sum(p_attn*value.unsqueeze(1), -2), p_attn
       
    def forward(self, x, time_steps):
        time_steps = time_steps.cpu()
        mask = x[:, :, self.dim:]
        mask = torch.cat((mask, mask), 2)
        if self.learn_emb:
            key = self.learn_time_embedding(time_steps).to(self.device)
            query = self.learn_time_embedding(self.query.unsqueeze(0)).to(self.device)
        else:
            key = self.fixed_time_embedding(time_steps).to(self.device)
            query = self.fixed_time_embedding(self.query.unsqueeze(0)).to(self.device) 
        out, _ = self.attention(query, key, x, mask)
        out = self.cross(out)
        if not self.use_classif:
            out, _ = self.gru_rnn(out)
            out = self.hiddens_to_z0(out)
        else:
            _, h = self.gru_rnn(out)
            out = self.classifier(h.squeeze(0))
        return out
    
    
class dec_rnn3(nn.Module):
 
    def __init__(self, input_dim, query, latent_dim=2, nhidden=16, 
                 embed_time=16,learn_emb=False, device='cuda'):
        super(dec_rnn3, self).__init__()
        self.embed_time = embed_time
        self.dim = input_dim
        self.device = device
        self.nhidden = nhidden
        self.query = query
        self.learn_emb = learn_emb
        self.gru_rnn = nn.GRU(latent_dim, nhidden, bidirectional=True, batch_first=True)
        self.linears = nn.ModuleList([nn.Linear(embed_time, embed_time), 
                                     nn.Linear(embed_time, embed_time),
                                     nn.Linear(2*nhidden, 2*nhidden)])
      
        if learn_emb:
            self.periodic = nn.Linear(1, embed_time-1)
            self.linear = nn.Linear(1, 1)
        self.z0_to_obs = nn.Sequential(
            nn.Linear(2*nhidden, 50),
            nn.ReLU(),
            nn.Linear(50, input_dim))
        
    def learn_time_embedding(self, tt):
        tt = tt.to(self.device)
        tt = tt.unsqueeze(-1)
        out2 = torch.sin(self.periodic(tt))
        out1 = self.linear(tt)
        return torch.cat([out1, out2], -1)
        
        
    def fixed_time_embedding(self, pos):
        d_model = self.embed_time
        pe = torch.zeros(pos.shape[0], pos.shape[1], d_model)
        position = 48.*pos.unsqueeze(2)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(np.log(10.0) / d_model))
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe
    
    def attention(self, query, key, value, mask=None, dropout=None):
        "Compute 'Scaled Dot Product Attention'"
        batch, seq_len, dim = value.size()
        d_k = query.size(-1)
        query, key, value = [l(x) for l, x in zip(self.linears, (query, key, value))]
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(d_k)
        scores = scores[:, :, :, None].repeat(1, 1, 1, dim)
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1) == 0, -1e9)
        p_attn = F.softmax(scores, dim = -2)
        if dropout is not None:
            p_attn = dropout(p_attn)
        #return torch.sum(p_attn*value, -2), p_attn
        return torch.sum(p_attn*value.unsqueeze(1), -2), p_attn
       
    def forward(self, z, time_steps):
        out, _ = self.gru_rnn(z)
        time_steps = time_steps.cpu()
        if self.learn_emb:
            query = self.learn_time_embedding(time_steps).to(self.device)
            key = self.learn_time_embedding(self.query.unsqueeze(0)).to(self.device)
        else:
            query = self.fixed_time_embedding(time_steps).to(self.device)
            key = self.fixed_time_embedding(self.query.unsqueeze(0)).to(self.device)
        out, _ = self.attention(query, key, out)
        out = self.z0_to_obs(out)
        return out

