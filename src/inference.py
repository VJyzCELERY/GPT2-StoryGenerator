import numpy as np
import torch
import torch.nn.functional as F
import hnswlib
from sentence_transformers import SentenceTransformer
import tiktoken
from dataclasses import dataclass
from src.model import GPT

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

    def generate_sequences(self, prompt, max_tokens=50, seed=42,longer_story=True, temperature=0.8, end_prob=0.1):
        self.model.eval()

        rag_context = ""
        if self.retriever is not None and self.embed_model is not None:
            query_emb = self.embed_model.encode([prompt])
            retrieved = self.retriever.search(query_emb, k=3)
            rag_context = "\n\n".join(retrieved)

        model_prompt = f"{rag_context}\n\n{prompt}" if rag_context else prompt

        tokens = self.token_encoder.encode(model_prompt)
        tokens = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(self.device)
        gen_tokens = tokens

        sample_rng = torch.Generator(device=self.device).manual_seed(seed)

        eos_id = self.token_encoder.encode(
            "<|endoftext|>", 
            allowed_special={"<|endoftext|>"}
        )[0]

        context_len = self.model.config.context_length

        while gen_tokens.shape[-1] <= max_tokens:

            if gen_tokens.shape[1] > context_len:
                gen_tokens = gen_tokens[:, -context_len:]

            idx_cond = gen_tokens

            with torch.no_grad():
                with torch.autocast(device_type=self.device_type, dtype=torch.bfloat16):
                    logits, loss = self.model(idx_cond)

            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)

            if longer_story:
                if gen_tokens.shape[1] < context_len + 100:
                    probs[:, eos_id] = 0
                else:
                    probs[:, eos_id] *= end_prob

            k = min(50, probs.shape[-1])
            topk_probs, topk_indices = torch.topk(probs, k, dim=-1)
            ix = torch.multinomial(topk_probs, 1, generator=sample_rng)
            next_tok = torch.gather(topk_indices, -1, ix)

            if next_tok.item() == eos_id:
                break

            gen_tokens = torch.cat([gen_tokens, next_tok], dim=1)

            yield self.token_encoder.decode([next_tok.item()])

        final_text = self.token_encoder.decode(gen_tokens[0, :max_tokens].tolist())
        yield final_text

    def print_stream(self, prompt, max_tokens=200,seed=42,longer_story=True, temperature=0.8, end_prob=0.1):
        text = prompt
        print(prompt,end=" ", flush=True)
        for piece in self.generate_sequences(prompt, max_tokens=max_tokens,seed=seed,longer_story=longer_story,temperature=temperature,end_prob=end_prob):
            text += piece
            print(piece, end="", flush=True)
        return text
        

