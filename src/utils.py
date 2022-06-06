import csv

from torch.utils.data import TensorDataset, DataLoader
from typing import List, Tuple

import wandb

import numpy as np
import torch
from torch.nn import CrossEntropyLoss

from sklearn.metrics import f1_score
from transformers import BertTokenizerFast, BertForSequenceClassification, BatchEncoding

from tqdm import tqdm
import os
from os.path import join, dirname
from sklearn.model_selection import train_test_split


def open_data(filename: str) -> Tuple[List[str], List[int]]:
    texts, labels = [], []
    with open(f'data/{filename}') as f:
        reader = csv.reader(f)
        _ = next(reader)
        for row in reader:
            texts.append(row[1].strip('"'))
            labels.append(int(row[2]))
    return texts, labels


def get_tokens(texts: List[str], tokenizer: BertTokenizerFast, max_length: int = 30) -> BatchEncoding:
    tokens = tokenizer.batch_encode_plus(
        texts,
        max_length=max_length,
        padding='max_length',
        truncation=True
    )
    return tokens


def get_dataloaders(tokens: BatchEncoding, labels: List[Tuple[int, int]], batch_size: int, seed: int = None) -> \
        Tuple[DataLoader, DataLoader]:
    train_indxs, val_indxs = train_test_split(list(range(len(labels))), train_size=0.75, shuffle=True,
                                              random_state=seed if seed is not None else wandb.config.seed)

    seq = torch.Tensor(tokens['input_ids']).long()
    mask = torch.Tensor(tokens['attention_mask'])
    y = torch.Tensor(labels)

    train_seq = seq[train_indxs]
    train_mask = mask[train_indxs]
    train_y = y[train_indxs]

    val_seq = seq[val_indxs]
    val_mask = mask[val_indxs]
    val_y = y[val_indxs]

    train_data = TensorDataset(train_seq, train_mask, train_y)
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    val_data = TensorDataset(val_seq, val_mask, val_y)
    val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    return train_dataloader, val_dataloader


def train_one_epoch(model, dataloader: DataLoader, optimizer: torch.optim, scheduler: torch.optim.lr_scheduler,
                    step: int, epoch: int, grad_accum: int, p_zero: float) -> int:
    device = model.device
    criterion = CrossEntropyLoss(weight=torch.tensor([p_zero, 1-p_zero], device=device))
    model.train()
    total_loss = 0
    for batch in tqdm(dataloader):
        step += 1
        batch = [el.to(device) for el in batch]
        sent_id, mask, labels = batch
        # loss = model(sent_id, mask, labels=labels).loss
        pred = model(sent_id, mask).logits
        loss = criterion(pred, labels)
        loss /= grad_accum
        loss.backward()
        total_loss += loss.item()
        if not step % grad_accum:
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            wandb.log({'epoch': epoch, 'loss': total_loss, 'lr': optimizer.param_groups[0]['lr']}, step=step)
            total_loss = 0
    return step


def evaluate(model: BertForSequenceClassification, dataloader: DataLoader) -> Tuple[float, float]:
    model.eval()
    device = model.device
    total_preds, gts = [], []
    total_loss = 0
    for batch in tqdm(dataloader):
        batch = [t.to(device) for t in batch]
        sent_id, mask, labels = batch
        gts.extend(torch.argmax(labels, dim=1).tolist())
        with torch.no_grad():
            preds = model(sent_id, mask, labels=labels)
            total_loss += preds.loss.item()
            preds = torch.argmax(preds.logits, dim=1).cpu().numpy()
            total_preds.append(preds)

    total_preds = np.concatenate(total_preds, axis=0)
    return f1_score(gts, total_preds, zero_division=0), total_loss / len(dataloader)


def save_model(model: BertForSequenceClassification, name: str):
    model_path = join('models', f'{name}.pt')
    os.makedirs(dirname(model_path), exist_ok=True)
    torch.save(model.state_dict(), model_path)
    wandb.save(model_path)
