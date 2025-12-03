import numpy as np
import torch
import torch.nn.functional as F
import hnswlib
from sentence_transformers import SentenceTransformer
import tiktoken
from dataclasses import dataclass
from src.model import GPT

def concat(prev, new):
    if prev and prev[-1].isalnum() and new and new[0].isalnum():
        return prev + " " + new
    return prev + new

class HNSWRetriever:
    def __init__(self, dim=768, space='cosine'):
        self.index = hnswlib.Index(space=space, dim=dim)
        self.initialized = False
        self.data = []
    
    def build(self, embeddings, texts, ef=200, M=48):
        """
        embeddings: numpy array (N, dim)
        texts: list of strings (N)
        """
        num_elements = embeddings.shape[0]
        self.data = texts

        self.index.init_index(max_elements=num_elements, ef_construction=ef, M=M)
        self.index.add_items(embeddings)
        self.index.set_ef(ef)

        self.initialized = True
        print("HNSW index built:", num_elements, "items")

    def search(self, query_emb, k=3):
        if not self.initialized:
            return []

        labels, distances = self.index.knn_query(query_emb, k=k)
        results = [self.data[idx] for idx in labels[0]]
        return results

class GPTInfer:
    def __init__(self, model, token_encoder, device, retriever=None, embed_model=None):
        self.model = model
        self.token_encoder = token_encoder
        self.device = device
        self.retriever = retriever     
        self.embed_model = embed_model 
        self.device_type = 'cuda' if device.startswith('cuda') else 'cpu'
    def get_token_length(self,text):
        return len(self.token_encoder.encode(text,allowed_special={"<|endoftext|>"}))
    def retrieve_context(self, query, k=3):
        if self.retriever is None or self.embed_model is None:
            return ""

        q_emb = self.embed_model.encode([query]) 

        retrieved_texts = self.retriever.search(q_emb, k=k)

        return "\n\n".join(retrieved_texts)
    
    def apply_frequency_penalty_and_blocking(
        self,
        logits,                 
        gen_tokens,            
        frequency_penalty=0.5,  
        no_repeat_ngram_size=3, 
    ):
        logits = logits.clone().float() 

        if frequency_penalty and frequency_penalty > 0.0:
            counts = {}
            for t in gen_tokens[0].tolist():
                counts[t] = counts.get(t, 0) + 1
            if counts:
                vocab_size = logits.shape[-1]
                penalty = torch.zeros(vocab_size, dtype=logits.dtype, device=logits.device)
                for tok, c in counts.items():
                    if 0 <= tok < vocab_size:
                        penalty[tok] = float(c) * float(frequency_penalty)
                logits = logits - penalty.unsqueeze(0)

        if no_repeat_ngram_size and no_repeat_ngram_size > 0:
            n = no_repeat_ngram_size
            cur = gen_tokens[0].tolist()
            if len(cur) >= n - 1:
                banned_next = set()
                for i in range(len(cur) - (n - 1)):
                    ngram = tuple(cur[i:i + n])
                    prefix = tuple(ngram[:-1])
                    banned_next.add(ngram[-1])
                last_prefix = tuple(cur[-(n - 1):]) if n > 1 else tuple()
                for i in range(len(cur) - (n - 1)):
                    if tuple(cur[i:i + (n - 1)]) == last_prefix and i + (n - 1) < len(cur):
                        banned_token = cur[i + (n - 1)]
                        if 0 <= banned_token < logits.shape[-1]:
                            logits[0, banned_token] = -1e9

        return logits
    def sample_next_token(
        self,
        logits,                 
        gen_tokens,             
        seed_rng,              
        temperature=0.8,
        top_k=None,
        top_p=0.9,
        repetition_penalty=1.2,
        frequency_penalty=0.5,
        no_repeat_ngram_size=3,
        recent_tokens_window=200,
    ):
        logits = logits.clone().float()

        recent = gen_tokens[0, -recent_tokens_window:].tolist()
        if repetition_penalty is not None and repetition_penalty != 1.0:
            for t in set(recent):
                if 0 <= t < logits.shape[-1]:
                    logits[0, t] /= float(repetition_penalty)

        logits = self.apply_frequency_penalty_and_blocking(
            logits,
            gen_tokens,
            frequency_penalty=frequency_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
        )

        if temperature is not None and temperature != 1.0:
            logits = logits / float(temperature)

        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        sorted_probs = F.softmax(sorted_logits, dim=-1)

        if top_k is not None:
            k = min(int(top_k), sorted_logits.shape[-1])
            sorted_logits = sorted_logits[:, :k]
            sorted_idx = sorted_idx[:, :k]
            sorted_probs = sorted_probs[:, :k]

        if top_p is not None and 0.0 < top_p < 1.0:
            cum_probs = torch.cumsum(sorted_probs, dim=-1)
            mask = cum_probs <= top_p
            if not mask.any():
                mask[0, 0] = True
            keep_count = int(mask.sum(dim=-1).item())
            sorted_probs = sorted_probs[:, :keep_count]
            sorted_idx = sorted_idx[:, :keep_count]

        sorted_probs = sorted_probs / (sorted_probs.sum(dim=-1, keepdim=True) + 1e-12)
        next_index_in_sorted = torch.multinomial(sorted_probs, 1, generator=seed_rng)  
        next_tok = sorted_idx.gather(-1, next_index_in_sorted)

        return int(next_tok.item())
    def generate_sequences(
        self,
        prompt,
        max_new_tokens=50,
        seed=42,
        longer_story=True,
        temperature=0.8,
        top_k=None,
        top_p=0.9,
        repetition_penalty=1.2,
        frequency_penalty=0.5,
        no_repeat_ngram_size=3,
        end_prob=0.1,
        rag_k=3,
        rag_delimiter="\n\n---\n\n",
    ):
        self.model.eval()

        rag_context = ""
        if self.retriever is not None and self.embed_model is not None and rag_k > 0:
            q_emb = self.embed_model.encode([prompt])
            retrieved = self.retriever.search(q_emb, k=rag_k)

            seen = set()
            dedup = []
            for r in retrieved:
                if r not in seen:
                    seen.add(r)
                    dedup.append(r)
            if dedup:
                rag_context = rag_delimiter.join(dedup)

        model_prompt = f"{rag_context}{rag_delimiter if rag_context else ''}{prompt}" if rag_context else prompt

        tokens = self.token_encoder.encode(model_prompt)
        tokens = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(self.device)
        gen_tokens = tokens.clone()

        sample_rng = torch.Generator(device=self.device).manual_seed(seed)
        eos_id = self.token_encoder.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})[0]
        context_len = self.model.config.context_length
        new_tokens_generated = 0
        HARD_MAX_TOTAL = context_len + max_new_tokens + 10

        while new_tokens_generated < max_new_tokens and gen_tokens.shape[1] < HARD_MAX_TOTAL:
            if gen_tokens.shape[1] > context_len:
                idx_cond = gen_tokens[:, -context_len:]
            else:
                idx_cond = gen_tokens

            with torch.no_grad():
                try:
                    with torch.autocast(device_type=self.device_type, dtype=torch.bfloat16):
                        logits, _ = self.model(idx_cond)
                except Exception:
                    logits, _ = self.model(idx_cond)

            next_logits = logits[:, -1:, :].squeeze(1)

            if longer_story and new_tokens_generated < 5:
                next_logits[0, eos_id] = next_logits[0, eos_id] / 4.0

            next_token_id = self.sample_next_token(
                logits=next_logits,
                gen_tokens=gen_tokens,
                seed_rng=sample_rng,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                frequency_penalty=frequency_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                recent_tokens_window=200,
            )

            if next_token_id == eos_id:
                break

            next_tok_tensor = torch.tensor([[next_token_id]], dtype=torch.long).to(self.device)
            gen_tokens = torch.cat([gen_tokens, next_tok_tensor], dim=1)
            new_tokens_generated += 1
            yield self.token_encoder.decode([next_token_id], errors='ignore')

        yield self.token_encoder.decode(gen_tokens[0, :].tolist(), errors='ignore')
    
    def print_stream(
        self,
        prompt,
        max_new_tokens=200,
        seed=42,
        longer_story=True,
        temperature=0.8,
        top_k=None,
        top_p=0.9,
        repetition_penalty=1.2,
        frequency_penalty=0.6,
        no_repeat_ngram_size=3,
        end_prob=0.1,
        rag_k=3,
    ):
        text = prompt
        last_piece = ""
        print(prompt, end="", flush=True)
        for piece in self.generate_sequences(
            prompt,
            max_new_tokens=max_new_tokens,
            seed=seed,
            longer_story=longer_story,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            frequency_penalty=frequency_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            end_prob=end_prob,
            rag_k=rag_k,
        ):
            if piece == last_piece:
                continue
            last_piece = piece
            text = concat(text, piece)
            print(piece, end="", flush=True)
        return text
        

