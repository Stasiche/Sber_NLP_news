import wandb

import transformers
from transformers import logging
from transformers import AutoTokenizer, BertForSequenceClassification

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import OneCycleLR

from tqdm import trange
import os

from src.utils import open_data, get_tokens, get_dataloaders, train_one_epoch, evaluate, save_model
from random import seed

wandb.init(
    project='Sber_NLP_news',
    # config={
    #     'batch_size': 100,
    #     'grad_accum': 1,
    #     'epochs': 50,
    #     'max_length': 64,
    #     'lr': 2e-5,
    #     'seed': 42,
    #
    #     'model': 'cointegrated/rubert-tiny2',
    #     'w_decay': 1e-5
    # }
    config={
            'batch_size': 10,
            'grad_accum': 8,
            'epochs': 60,
            'max_length': 64,
            'lr': 1e-5,
            'seed': 42,
            'model': 'DeepPavlov/rubert-base-cased',
            'w_decay': 5e-6,
            'balanced': True
        }
)

logging.set_verbosity_error()
config = wandb.config
torch.manual_seed(config.seed)
seed(config.seed)
device = torch.device('cuda')
os.makedirs('models', exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained(config.model)
model = BertForSequenceClassification.from_pretrained(config.model, vocab_size=tokenizer.vocab_size, num_labels=2)

model.to(device)

texts, labels = open_data('train_data.csv')
labels = [(0, 1) if el else (1, 0) for el in labels]
tokens = get_tokens(texts, tokenizer, max_length=config.max_length)

train_dataloader, val_dataloader = get_dataloaders(tokens, labels, config.batch_size)

if config.balanced:
    p_zero = 0
    for _, _, labels in train_dataloader:
        p_zero += (labels == torch.tensor([1, 0])).sum(0)[0].item()
    p_zero /= len(train_dataloader.dataset)
else:
    p_zero = None

optimizer = Adam(model.parameters(), lr=config.lr, weight_decay=config.w_decay)

scheduler = transformers.get_cosine_schedule_with_warmup(optimizer,
                                                         num_warmup_steps=0.1 * len(
                                                             train_dataloader) * config.epochs // config.grad_accum,
                                                         num_training_steps=len(
                                                             train_dataloader) * config.epochs // config.grad_accum)

step = 0
min_eval, max_score = 0, 0
for epoch in trange(config.epochs):
    step = train_one_epoch(model, train_dataloader, optimizer, scheduler, step, epoch, config.grad_accum, p_zero)
    score, eval_loss = evaluate(model, val_dataloader)

    wandb.log({'score': score, 'eval_loss': eval_loss}, step=step)
    if min_eval > eval_loss:
        save_model(model, 'min_eval')
        min_eval = eval_loss
    if max_score < score:
        save_model(model, 'max_score')
        max_score = score
