import torch
import torch.nn as nn
import torch.nn.functional as f
from dataclasses import dataclass
import time
import os
from src.model import GPT,Config
import numpy as np
import math


torch.set_float32_matmul_precision('high')

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
        for step in range(max_steps):
            t0 = time.time()
            self.is_last_step = (step == max_steps-1)

            if step % self.eval_freq == 0 or self.is_last_step:
                self.evaluate_validation(step)

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
            if self.master_process:
                print(f'step {step:4d} | loss: {batch_loss.item():.6f} | lr: {lr:.2e} | norm: {norm:.4f} | dt: {dt:.4f}ms | tok/sec: {tokens_per_sec:.4f}')
                with open(self.logpath, 'a') as f:
                    f.write(f'{step} train {batch_loss.item():.6f}\n')
    
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
            print(f'Val loss: {val_loss_accum.item():.4f}')
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
    
    def estimate_lr(self, step, warmup_steps, max_steps, max_lr, min_lr):
        if step < warmup_steps:
            return max_lr * (step+1) / warmup_steps
        if step > max_steps:
            return min_lr
        decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return min_lr + coeff * (max_lr - min_lr)
