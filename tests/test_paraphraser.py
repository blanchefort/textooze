import json

from textooze.augmenters.paraphraser import paraphraser

with open('tests/data/en_short_texts.json') as fp:
    data = json.load(fp)
    en_texts = data['texts']
    en_labels = data['labels']


class TestParaphraserEn:
    en_texts = en_texts[:5]
    en_labels = en_labels[:5]
    en_big_text = ' '.join(en_texts)

    def test_translation(self):
        new_texts, new_labels = paraphraser(
            texts=self.en_texts,
            labels=self.en_labels,
            num_beams=3,
            num_return_sequences=3,
            max_length=30
        )
        assert len(new_texts) > 0
        assert len(new_labels) == len(new_texts)

    def test_new_texts(self):
        new_texts, _ = paraphraser(
            texts=self.en_texts,
            labels=self.en_labels,
            num_beams=3,
            num_return_sequences=3,
            max_length=30
        )
        for text in new_texts:
            assert text not in self.en_texts

    def test_new_labels(self):
        _, new_labels = paraphraser(
            texts=self.en_texts,
            labels=self.en_labels,
            num_beams=3,
            num_return_sequences=3,
            max_length=30
        )
        for label in new_labels:
            assert label in self.en_labels

    def test_big_text(self):
        new_texts, new_labels = paraphraser(
            texts=self.en_texts,
            labels=self.en_labels,
            num_beams=3,
            num_return_sequences=3,
            max_length=30
        )
        assert len(new_texts) > 0
        assert len(new_labels) == len(new_texts)
