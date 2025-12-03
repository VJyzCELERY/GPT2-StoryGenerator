import torch
import torch.nn as nn
import torch.nn.functional as f
from dataclasses import dataclass
import time
import os
from src.model import GPT,Config
from sacrebleu import corpus_bleu
from rouge_score import rouge_scorer
import numpy as np
import math

torch.set_float32_matmul_precision('high')

def repetition_rate(text, n=3):
    tokens = text.split()
    if len(tokens) < n:
        return 0.0
    ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
    return 1 - (len(set(ngrams)) / len(ngrams))

def distinct_n(text, n=1):
    tokens = text.split()
    if len(tokens) < n:
        return 0.0
    
    ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
    return len(set(ngrams)) / len(ngrams)


def compute_self_bleu(generated_texts):
    if len(generated_texts) < 2:
        return 0.0

    scores = []
    N = len(generated_texts)

    for i in range(N):
        hyp = generated_texts[i]
        refs = generated_texts[:i] + generated_texts[i+1:]

        bleu = corpus_bleu([hyp], [refs]).score
        scores.append(bleu)

    return sum(scores) / len(scores)

class Trainer:
    def __init__(self,model : GPT,optimizer,train_loader,val_loader,token_encoder,eval_freq,grad_accum_steps,device,master_process,logpath):
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.token_encoder = token_encoder
        self.master_process = master_process
        self.eval_freq = eval_freq
        self.grad_accum_steps = grad_accum_steps
        self.device = device
        self.device_type = 'cuda' if device.startswith('cuda') else 'cpu'
        self.logpath=logpath
    
    def train(self,max_steps,warmup_steps,max_lr,min_lr):
        history={
            'val_losses':[],
            'perplexities':[],
            'train_losses':[]
        }
        for step in range(max_steps):
            val_loss = None
            perplexity=None
            t0 = time.time()
            self.is_last_step = (step == max_steps-1)
            self.model.train()
            self.optimizer.zero_grad()
            batch_loss = 0.0

            for mini_step in range(self.grad_accum_steps):
                inp, target = self.train_loader.next_batch()
                inp, target = inp.to(self.device),target.to(self.device)

                with torch.autocast(device_type=self.device_type,dtype=torch.bfloat16):
                    logits,loss = self.model(inp,target)
                loss /=self.grad_accum_steps
                batch_loss+=loss.detach()
                loss.backward()
            norm = nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            lr = self.estimate_lr(step,warmup_steps,max_steps,max_lr,min_lr)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            self.optimizer.step()
            if self.device_type == 'cuda':
                torch.cuda.synchronize()
            dt = (time.time() - t0) * 1000.0    # in ms
            tokens_processed = self.train_loader.B * self.train_loader.T * self.grad_accum_steps * 1
            tokens_per_sec = tokens_processed / dt
            
            if step % self.eval_freq == 0 or self.is_last_step:
                val_loss,perplexity = self.evaluate_validation(step)
                history['val_losses'].append(val_loss)
                history['perplexities'].append(perplexity)
            
            history['train_losses'].append(batch_loss.item())
            if self.master_process:
                print(f'step {step:4d} | train loss: {batch_loss.item():.2f}{f' | val loss: {val_loss:.2f}' if val_loss is not None else ''}{f' | perplexity: {perplexity:.2f}' if perplexity is not None else ''} | lr: {lr:.2e} | norm: {norm:.4f} | dt: {dt:.4f}ms | tok/sec: {tokens_per_sec:.4f}')
                with open(self.logpath, 'a') as f:
                    f.write(f'{step} train {batch_loss.item():.6f}\n')

        evaluation =self.evaluate_text_metrics(
            max_samples=60,
            gen_len=128,
            do_sample=False,    
            top_k=None,
            temperature=0.2,
            eos_token_id=None    
        )
        return history,evaluation
    
    def evaluate_text_metrics(self, max_samples=100, gen_len=50, do_sample=False, top_k=None, temperature=1.0, eos_token_id=None):
        self.model.eval()
        self.val_loader.reset()

        hyps = []
        refs = []
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

        samples_collected = 0
        while samples_collected < max_samples:
            try:
                inp, target = self.val_loader.next_batch()
            except StopIteration:
                break

            inp = inp.to(self.device)
            target = target.to(self.device)

            if inp.shape[1] > self.model.config.context_length:
                inp = inp[:, -self.model.config.context_length:]

            with torch.no_grad():
                generated = self.model.generate(
                    inp,
                    max_new_tokens=gen_len,
                    temperature=temperature,
                    top_k=top_k,
                    do_sample=do_sample,
                    eos_token_id=eos_token_id
                )

            B = generated.shape[0]
            for i in range(B):
                gen_ids = generated[i, inp.shape[1]:].tolist()

                pred_text = self.token_encoder.decode(gen_ids)
                ref_text  = self.token_encoder.decode(target[i].tolist())

                hyps.append(pred_text)
                refs.append(ref_text)
                samples_collected += 1
                if samples_collected >= max_samples:
                    break

        if len(hyps) == 0:
            return 0.0, 0.0

        rep_scores = []
        distinct1_scores = []
        distinct2_scores = []

        for txt in hyps:
            rep_scores.append(repetition_rate(txt, n=3))
            distinct1_scores.append(distinct_n(txt, n=1))
            distinct2_scores.append(distinct_n(txt, n=2))

        avg_rep = sum(rep_scores) / len(rep_scores)
        avg_d1 = sum(distinct1_scores) / len(distinct1_scores)
        avg_d2 = sum(distinct2_scores) / len(distinct2_scores)


        bleu = corpus_bleu(hyps, [refs]).score
        self_bleu=compute_self_bleu(hyps)
        rouge_scores = []
        for h, r in zip(hyps, refs):
            sc = scorer.score(r, h)['rougeL'].fmeasure
            rouge_scores.append(sc)
        rouge_l = sum(rouge_scores) / len(rouge_scores)

        if self.master_process:
            print(f"[Text Eval] samples={len(hyps)} BLEU={bleu:.2f} ROUGE-L={rouge_l:.4f} SELF-BLEU={self_bleu:.2f} REP={avg_rep:.4f} D1={avg_d1:.4f} D2={avg_d2:.4f}")
            with open(self.logpath, 'a') as f:
                f.write(f"eval samples={len(hyps)} BLEU={bleu:.2f} ROUGE-L={rouge_l:.4f} SELF-BLEU={self_bleu:.2f} REP={avg_rep:.4f} D1={avg_d1:.4f} D2={avg_d2:.4f}\n")

        return {"bleu":bleu,"rogue-l":rouge_scores,"self-bleu":self_bleu,"repetition":rep_scores,"D1":distinct1_scores,"D2":distinct2_scores}

    def evaluate_validation(self,step):
        self.model.eval()
        self.val_loader.reset()
        with torch.no_grad():
            val_loss_accum = 0.0
            val_steps = 20
            for _ in range(val_steps):
                inp, target = self.val_loader.next_batch()
                inp, target = inp.to(self.device),target.to(self.device)

                with torch.autocast(device_type=self.device_type,dtype=torch.bfloat16):
                    logits,loss = self.model(inp,target)
                loss /=val_steps
                val_loss_accum+=loss.detach()

        if self.master_process:
            perplexity = math.exp(val_loss_accum.item())
            with open(self.logpath, 'a') as f:
                f.write(f'{step} val {val_loss_accum.item():.4f}\n')

            if step > 0 and (step % 10000 == 0 or self.is_last_step):
                raw_model = self.model
                logdir = os.path.dirname(self.logpath)
                ckpt_path = os.path.join(logdir, f'model_{step:05d}.pt')
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'config': raw_model.config,
                    'step': step,
                    'val_loss': val_loss_accum.item()
                }
                torch.save(checkpoint, ckpt_path)
        return val_loss_accum.item(),perplexity
    
    def estimate_lr(self, step, warmup_steps, max_steps, max_lr, min_lr):
        if step < warmup_steps:
            return max_lr * (step+1) / warmup_steps
        if step > max_steps:
            return min_lr
        decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return min_lr + coeff * (max_lr - min_lr)
