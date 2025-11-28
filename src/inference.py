import numpy as np
import torch
import torch.nn.functional as F
import tiktoken
from dataclasses import dataclass
from src.model import GPT

class GPTInfer:
    def __init__(self,model,token_encoder,device):
        self.model = model
        self.token_encoder = token_encoder
        self.device = device
        self.device_type = 'cuda' if device.startswith('cuda') else 'cpu'
    def get_token_length(self,text):
        return len(self.token_encoder.encode(text,allowed_special={"<|endoftext|>"}))
    def generate_sequences(self,prompt,max_tokens=50,seed=42,longer_story=True,temperature=0.8,end_prob=0.1):
        self.model.eval()
        tokens = self.token_encoder.encode(prompt)
        tokens = torch.tensor(tokens,dtype=torch.long)
        tokens = tokens.unsqueeze(0)
        gen_tokens = tokens.to(self.device)
        sample_rng = torch.Generator(device=self.device).manual_seed(seed)
        eos_id = self.token_encoder.encode("<|endoftext|>",allowed_special={"<|endoftext|>"})[0]
        print(f'{prompt}',end='')
        context_len = self.model.config.context_length
        summary_len = 100
        while gen_tokens.shape[-1] <= max_tokens:
            # idx_cond = gen_tokens[:, -context_len:]
            if gen_tokens.shape[1] > context_len:
                idx_cond = torch.cat([gen_tokens[:, :summary_len], 
                                    gen_tokens[:, - (context_len - summary_len):]], dim=1)
            else:
                idx_cond = gen_tokens
            with torch.no_grad():
                with torch.autocast(device_type=self.device_type, dtype=torch.bfloat16):
                    logits, loss = self.model(idx_cond)  
                logits = logits[:, -1, :]   
                logits = logits / temperature
                probs = F.softmax(logits, dim=-1)
                # Prevent EOS so we can generate longer tokens might reduce coherence or cause hallucination
                # if longer_story:
                #     probs[:, eos_id] = 0.0 
                #     probs = probs / probs.sum(dim=-1, keepdim=True)
                if longer_story:
                    min_story_len = context_len+100
                    if gen_tokens.shape[1] < min_story_len:
                        probs[:, eos_id] = 0.0
                    else:
                        # let EOS probability gradually appear
                        probs[:, eos_id] *= end_prob

                k = min(50, probs.shape[-1])
                topk_probs, topk_indices = torch.topk(probs, k, dim=-1) 
                ix = torch.multinomial(topk_probs, num_samples=1, generator=sample_rng)
                next_tok = torch.gather(topk_indices, -1, ix)   
                if next_tok.item() == eos_id:
                    break
                gen_tokens = torch.cat([gen_tokens, next_tok], dim=1)
                print(self.token_encoder.decode([next_tok]),end="")
        tokens = gen_tokens[0,:max_tokens].tolist()
        gen_text = self.token_encoder.decode(tokens)
        return gen_text
    

