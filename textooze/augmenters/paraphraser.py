"""Аугментация методом перефразировки."""
from typing import List, Tuple

import torch
import tqdm
from transformers import PegasusForConditionalGeneration, PegasusTokenizer


def paraphraser(
    texts: List,
    labels: List,
    num_beams: int = 10,
    num_return_sequences: int = 3,
    max_length: int = 60,
    model_name: str = 'tuner007/pegasus_paraphrase') \
        -> Tuple[List, List]:
    """парафразер Pegasus.

    Args:
        texts (List): Тексты для перефразировки.
        labels (List): Метки текстов.
        num_beams (int, optional): Количество лучей для генерации новой последовательности. Defaults to 10.
        num_return_sequences (int, optional): Количество новых текстов, которое необходимо
        сгенерировать для каждого сэмпла. Defaults to 3.
        max_length (int, optional): Максимальна длина входящего и генерируемного текста (токенов). Defaults to 60.
        model_name (str, optional): Название модели в хабе transformers. Defaults to 'tuner007/pegasus_paraphrase'.

    Returns:
        [type]: Два списка - новые тексты и их метки.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    tokenizer = PegasusTokenizer.from_pretrained(model_name)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        model = PegasusForConditionalGeneration.from_pretrained(model_name).to(device)
    else:
        model = PegasusForConditionalGeneration.from_pretrained(model_name)

    new_texts, new_labels = [], []
    for text, label in tqdm.tqdm(zip(texts, labels), total=len(texts), desc='paraphraser'):
        tokenized_data = tokenizer(
            [text],
            truncation=True,
            padding='longest',
            max_length=max_length,
            return_tensors="pt")
        translated = model.generate(**tokenized_data.to(device),
                                    max_length=max_length,
                                    num_beams=num_beams,
                                    num_return_sequences=num_return_sequences,
                                    temperature=1.5)
        tgt_texts = tokenizer.batch_decode(translated, skip_special_tokens=True)
        for t in set(tgt_texts):
            if t != text:
                new_texts.append(t)
                new_labels.append(label)

    return new_texts, new_labels
