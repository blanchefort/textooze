from typing import List, Tuple
import random
import tqdm
import torch
from transformers import pipeline, BertTokenizerFast


def contextual_replacer(texts: List, labels: List, top_k: int, model_name: str = 'bert-base-uncased') -> Tuple[List, List]:
    """Контекстная замена одного случайного токена.

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
        fill_mask = pipeline('fill-mask', model=model_name, device=0)
    else:
        fill_mask = pipeline('fill-mask', model=model_name)
    tokenizer = BertTokenizerFast.from_pretrained(model_name)

    new_texts, new_labels = [], []
    for text, label in tqdm.notebook.tqdm(zip(texts, labels), total=len(texts), desc='contextual_replacer'):
        tokens = tokenizer.tokenize(text)[:510]
        if len(tokens) < 2:
            continue
        masked_token_pos = random.randint(0, len(tokens)-1)
        masked_token = tokens[masked_token_pos]
        tokens[masked_token_pos] = '[MASK]'
        counter = 0
        for item in fill_mask(" ".join(tokens), top_k=top_k+1):
            if item['token_str'] != masked_token:
                new_texts.append(item['sequence'])
                new_labels.append(label)
                counter += 1
            if counter == top_k:
                break
    return new_texts, new_labels
