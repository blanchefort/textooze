import json

from textooze.augmenters.contextual_imputer import contextual_imputer

with open('tests/data/ru_short_texts.json') as fp:
    data = json.load(fp)
    ru_texts = data['texts']
    ru_labels = data['labels']


class TestContextualImputer:
    ru_texts = ru_texts[:5]
    ru_labels = ru_labels[:5]
    ru_big_text = ' '.join(ru_texts)

    def test_translation(self):
        new_texts, new_labels = contextual_imputer(
            texts=self.ru_texts,
            labels=self.ru_labels,
            top_k=3
        )
        assert len(new_texts) > 0
        assert len(new_labels) == len(new_texts)

    def test_new_texts(self):
        new_texts, _ = contextual_imputer(
            texts=self.ru_texts,
            labels=self.ru_labels,
            top_k=3
        )
        for text in new_texts:
            assert text not in self.ru_texts

    def test_new_labels(self):
        _, new_labels = contextual_imputer(
            texts=self.ru_texts,
            labels=self.ru_labels,
            top_k=3
        )
        for label in new_labels:
            assert label in self.ru_labels

    def test_big_text(self):
        new_texts, new_labels = contextual_imputer(
            texts=self.ru_texts,
            labels=self.ru_labels,
            top_k=3
        )
        assert len(new_texts) > 0
        assert len(new_labels) == len(new_texts)
