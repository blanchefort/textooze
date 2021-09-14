import json

from textooze.augmenters.back_translation import back_translation

with open('tests/data/ru_short_texts.json') as fp:
    data = json.load(fp)
    ru_texts = data['texts']
    ru_labels = data['labels']


class TestBackTranslation:
    ru_texts = ru_texts[:5]
    ru_labels = ru_labels[:5]
    ru_big_text = ' '.join(ru_texts)

    def test_translation(self):
        new_texts, new_labels = back_translation(
            texts=self.ru_texts,
            labels=self.ru_labels,
            src='ru',
            tgt='en',
            batch_size=2)
        assert len(new_texts) > 0
        assert len(new_labels) == len(new_texts)

    def test_new_texts(self):
        new_texts, _ = back_translation(
            texts=self.ru_texts,
            labels=self.ru_labels,
            src='ru',
            tgt='en',
            batch_size=2)
        for text in new_texts:
            assert text not in self.ru_texts

    def test_new_labels(self):
        _, new_labels = back_translation(
            texts=self.ru_texts,
            labels=self.ru_labels,
            src='ru',
            tgt='en',
            batch_size=2)
        for label in new_labels:
            assert label in self.ru_labels

    def test_big_text(self):
        new_texts, new_labels = back_translation(
            texts=[self.ru_big_text],
            labels=[1],
            src='ru',
            tgt='en',
            batch_size=2)
        assert len(new_texts) > 0
        assert len(new_labels) == len(new_texts)
