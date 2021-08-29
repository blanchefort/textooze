from typing import List, Tuple
import tqdm
import torch
from transformers import pipeline


def back_translation(texts: List, labels: List, srs: str = 'en', tgt: str = 'ru', batch_size: int = 20) -> Tuple[List, List]:
    """Аументация методом обратного перевода.

    Args:
        texts (List): Список текстов для аугментации.
        labels (List): Список меток текстов.
        srs (str): Язык с такого переводить (язык текстов для аугментации).
        tgt (str): Язык, на который переводить.
        batch_size (int): Размер батча.

    Returns:
        [type]: Два списка - новые тексты и их метки.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    if torch.cuda.is_available():
        translator = pipeline(task='translation', model=f'Helsinki-NLP/opus-mt-{srs}-{tgt}', device=0)
        translator_back = pipeline(task='translation', model=f'Helsinki-NLP/opus-mt-{tgt}-{srs}', device=0)
    else:
        translator = pipeline(task='translation', model=f'Helsinki-NLP/opus-mt-{srs}-{tgt}')
        translator_back = pipeline(task='translation', model=f'Helsinki-NLP/opus-mt-{tgt}-{srs}')

    new_texts, new_labels = [], []
    for step in tqdm.notebook.trange(0, len(texts), batch_size, desc='back_translation'):
        transstep = False
        try:
            transleted = [t['translation_text'] for t in translator(texts[step:step+batch_size], max_length=460)]
            back = [t['translation_text'] for t in translator_back(transleted, max_length=460)]
            transstep = True
        except:
            continue
        if transstep == True:
            for t, b, l in zip(texts[step:step+batch_size], back, labels[step:step+batch_size]):
                if t != b:
                    new_texts.append(b)
                    new_labels.append(l)
    return new_texts, new_labels
