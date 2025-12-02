import torch
import torch.nn as nn
import torch.nn.functional as f
from dataclasses import dataclass
import inspect

@dataclass
class Config:
    context_length : int = 1024
    vocab_size: int = 50257
    num_layers : int = 12
    embedding_dim : int = 768
    num_heads: int = 12

class MultiHeadAttention(nn.Module):
    def __init__(self,config : Config,masked=False):
        super(MultiHeadAttention,self).__init__()
        self.num_heads = config.num_heads
        self.masked = masked
        self.embedding_dim = config.embedding_dim
        self.c_attention = nn.Linear(config.embedding_dim,3*config.embedding_dim)
        self.c_projection = nn.Linear(config.embedding_dim,config.embedding_dim)
        self.c_projection.SCALE_INIT = 1.0
    
    def forward(self,x):
        B, T, C = x.shape
        QKV = self.c_attention(x)
        Query_q,Key_k,Value_v = QKV.split(self.embedding_dim,dim=-1)
        Query_q = Query_q.view(B,T,self.num_heads,self.embedding_dim//self.num_heads).transpose(1,2)
        Key_k = Key_k.view(B,T,self.num_heads,self.embedding_dim//self.num_heads).transpose(1,2)
        Value_v = Value_v.view(B,T,self.num_heads,self.embedding_dim//self.num_heads).transpose(1,2)

        # out = f.scaled_dot_product_attention(Query_q,Key_k,Value_v,is_causal=True)
        if self.masked:
            out = f.scaled_dot_product_attention(Query_q,Key_k,Value_v,is_causal=True)
        else:
            out = f.scaled_dot_product_attention(Query_q,Key_k,Value_v,is_causal=False)
        out = out.transpose(1,2).contiguous().view(B,T,C)
        return self.c_projection(out)
    
class MLP(nn.Module):
    def __init__(self,config : Config):
        super(MLP,self).__init__()
        self.c_fc = nn.Linear(config.embedding_dim,4*config.embedding_dim)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_projection = nn.Linear(4*config.embedding_dim,config.embedding_dim)
        self.c_projection.SCALE_INIT = 1.0
    def forward(self,x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_projection(x)
        return x
    
class DecoderBlock(nn.Module):
    def __init__(self,config : Config):
        """Decoder block without the encoder output"""
        super(DecoderBlock,self).__init__()
        self.masked_attention = MultiHeadAttention(config,masked=True)
        self.layer_norm1 = nn.LayerNorm(config.embedding_dim)
        # self.attention = MultiHeadAttention(config,masked=False)
        # self.layer_norm2 = nn.LayerNorm(config.embedding_dim)
        self.mlp = MLP(config)
        self.layer_norm3 = nn.LayerNorm(config.embedding_dim)
    
    def forward(self,x):
        x = x + self.masked_attention(self.layer_norm1(x))
        # x = x + self.attention(self.layer_norm2(x))
        x = x + self.mlp(self.layer_norm3(x))
        return x

class TransformerDecoder(nn.Module):
    def __init__(self,config : Config):
        super(TransformerDecoder,self).__init__()
        self.config = config
        self.word_token_embedding = nn.Embedding(self.config.vocab_size,self.config.embedding_dim)
        self.word_position_embedding = nn.Embedding(self.config.context_length,self.config.embedding_dim)
        layers = [DecoderBlock(config) for _ in range(config.num_layers)]
        self.hidden_layers = nn.Sequential(*layers)
        self.layer_norm = nn.LayerNorm(self.config.embedding_dim)
    
    def forward(self,idx):
        B,T = idx.shape
        pos = torch.arange(0,T,dtype=torch.long,device=idx.device)
        pos_embed = self.word_position_embedding(pos)
        token_embed = self.word_token_embedding(idx)
        x = pos_embed + token_embed
        x = self.hidden_layers(x)
        x = self.layer_norm(x)
        return x

class GPT(nn.Module):
    def __init__(self,config : Config):
        super(GPT,self).__init__()
        self.config=config
        self.transformerDecoder = TransformerDecoder(config)
        self.language_modeling_head = nn.Linear(config.embedding_dim,config.vocab_size,bias=False)
        self.transformerDecoder.word_token_embedding.weight = self.language_modeling_head.weight
        self.apply(self._init_weights)
    
    def _init_weights(self,module):
        if isinstance(module,nn.Linear):
            std=0.02
            if hasattr(module,'SCALE_INIT'):
                std /= (2*self.config.num_layers)**0.5
            torch.nn.init.normal_(module.weight,mean=0,std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module,nn.Embedding):
            torch.nn.init.normal_(module.weight,mean=0,std=0.02)
    
    def forward(self,idx,targets=None):
        x = self.transformerDecoder(idx)
        logits = self.language_modeling_head(x)
        loss = None
        if targets is not None:
            loss = f.cross_entropy(logits.view(-1,logits.shape[-1]),targets.view(-1))
        return logits,loss  
    @torch.no_grad()
    def generate(self, idx, max_new_tokens=50, temperature=0.8, top_k=None, do_sample=False, eos_token_id=None):
        self.eval()

        B, T = idx.shape
        device = idx.device
        context_len = self.config.context_length

        if T > context_len:
            idx = idx[:, -context_len:]
            T = idx.shape[1]

        generated = idx.clone()

        for _ in range(max_new_tokens):
            input_ids = generated[:, -context_len:]

            logits, _ = self.forward(input_ids, targets=None) 
            next_logits = logits[:, -1, :] 

            if temperature != 1.0 and temperature > 0.0:
                next_logits = next_logits / temperature

            if do_sample:
                if top_k is not None and top_k > 0:
                    vals, idxs = next_logits.topk(top_k, dim=-1)
                    min_vals = vals[:, -1].unsqueeze(-1) 
                    mask = next_logits < min_vals
                    next_logits = next_logits.masked_fill(mask, float('-inf'))

                probs = torch.softmax(next_logits, dim=-1) 
                next_token = torch.multinomial(probs, num_samples=1)  
            else:
                next_token = torch.argmax(next_logits, dim=-1, keepdim=True) 

            generated = torch.cat([generated, next_token], dim=1)  

            if eos_token_id is not None:
                if (generated == eos_token_id).any(dim=1).all():
                    break

        return generated
    def configure_optimizer(self,weight_decay,lr,device_type,master_process):
        param_dict = {pn:p for pn, p in self.named_parameters() if p.requires_grad}

        decay_params = [p for pn, p in param_dict.items() if p.dim() >=2]
        nodecay_params = [p for pn, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params':decay_params,'weight_decay':weight_decay},
            {'params':nodecay_params,'weight_decay':0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        if master_process:
            print(f'num decay parameter tensors: {len(decay_params)} with {num_decay_params:,} parameters')
            print(f'num nodecay parameter tensors: {len(nodecay_params)} with {num_nodecay_params:,} parameters')
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        if master_process:
            print(f'using fused AdamW optimizer: {use_fused}')
        optimizer = torch.optim.AdamW(optim_groups, lr=lr, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer
