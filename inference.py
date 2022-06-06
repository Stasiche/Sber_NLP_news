import numpy as np
import torch

from transformers import logging
from transformers import BertTokenizerFast, BertForSequenceClassification

import os
import csv
from typing import List, Dict
from tqdm.auto import trange

from src.utils import open_data, get_tokens, get_dataloaders, evaluate


def concat_shorts(texts: List[str]) -> List[str]:
    res = [[texts[0]]]
    for el in texts[1:]:
        if len(el.split()) <= 5:
            res[-1].append(el)
        else:
            res.append([el])
    res = [''.join(el) for el in res]
    return res


def open_data_sub(filename: str) -> Dict[int, List[str]]:
    res = {}
    with open(f'data/{filename}') as f:
        reader = csv.reader(f)
        _ = next(reader)
        for row in reader:
            title_text = row[2] + ' ' + row[3]
            splits = [el.strip() + '. ' for el in title_text.split('. ')]
            res[int(row[1])] = concat_shorts(splits)
    return res


def write_submit(inp_filename: str, out_filename: str, probs: Dict[int, Dict[str, float]]) -> None:
    with open(f'data/{inp_filename}') as inp:
        with open(f'data/{out_filename}', 'w') as out:
            reader = csv.reader(inp)
            writer = csv.writer(out)

            writer.writerow(next(reader) + ['prob_mean', 'prob_mean_per_pos', 'prob_max'])
            for row in reader:
                if probs.get(int(row[1])):
                    writer.writerow(row + [str(round(val, 4)) for val in probs[int(row[1])].values()])


def inference():
    torch.manual_seed(0)
    logging.set_verbosity_error()
    model_path = 'models/max_score.pt'
    # device = torch.device('cpu')
    device = torch.device('cuda')
    os.makedirs('models', exist_ok=True)
    model_name = 'DeepPavlov/rubert-base-cased'
    tokenizer = BertTokenizerFast.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name, vocab_size=tokenizer.vocab_size, num_labels=2)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.to(device)
    model.eval()

    # eval model
    texts, labels = open_data('train_data.csv')
    labels = [(0, 1) if el else (1, 0) for el in labels]
    tokens = get_tokens(texts, tokenizer, max_length=64)

    _, val_dataloader = get_dataloaders(tokens, labels, 50, seed=42)
    score, eval_loss = evaluate(model, val_dataloader)

    print(f'Score: {score}, loss: {eval_loss}')

    # get submission
    grouped_texts = open_data_sub('test_data.csv')
    grouped_tokens = {i: tokenizer.batch_encode_plus(
        sent,
        max_length=64,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    ) for i, sent in grouped_texts.items()}

    device = model.device
    results = {}

    for i in trange(len(grouped_tokens)):
        sent_ids = grouped_tokens[i]['input_ids'].to(device)
        masks = grouped_tokens[i]['attention_mask'].to(device)
        with torch.no_grad():
            preds = model(sent_ids, masks)
            probs = torch.softmax(preds.logits, dim=1)[:, 1]
            prob_mean = probs.mean().cpu().item()
            prob_mean_per_pos = np.nanmax(np.hstack([probs[probs > 0.5].mean().cpu().item(), 0]))
            prob_max = probs.max().cpu().item()
            results[i] = {'prob_mean': prob_mean, 'prob_mean_per_pos': prob_mean_per_pos, 'prob_max': prob_max}

    write_submit('test_data.csv', 'submission.csv', results)
