"""Аугментация методом контекстной вставки случайного токена."""
import random
from typing import List, Tuple

import torch
import tqdm
from transformers import BertTokenizerFast, pipeline


def contextual_imputer(
    texts: List,
    labels: List,
    top_k: int,
    model_name: str = 'bert-base-uncased') \
        -> Tuple[List, List]:
    """Контекстная вставка одного нового случайного токена.

    Args:
        texts (List): Список входящих текстов.
        labels (List): Список меток текстов.
        top_k (int): Топ-k замен выбрать для аугментации каждого сэмпла.
        model_name (str): Название модели, расположенной в хабе transformers.

    Returns:
        [type]: Два списка - новые тексты и их метки.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        fill_mask = pipeline('fill-mask', model=model_name, device=0)
    else:
        fill_mask = pipeline('fill-mask', model=model_name)
    tokenizer = BertTokenizerFast.from_pretrained(model_name)

    new_texts, new_labels = [], []
    for text, label in tqdm.tqdm(zip(texts, labels), total=len(texts), desc='contextual_imputer'):
        tokens = tokenizer.tokenize(text)[:509]
        if len(tokens) < 3:
            continue
        masked_token_pos = random.randint(0, len(tokens)-2)
        tokens = tokens[:masked_token_pos] + ['[MASK]'] + tokens[masked_token_pos:]
        for item in fill_mask(" ".join(tokens), top_k=top_k):
            new_texts.append(item['sequence'])
            new_labels.append(label)
    return new_texts, new_labels
